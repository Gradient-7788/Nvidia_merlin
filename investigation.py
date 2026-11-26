"""
NVIDIA Merlin-inspired Recommendation System - Investigation & Benchmarking Engine

Every number this script reports is computed from trained models on held-out data.
Nothing is hardcoded.

Topics:
1. Popularity baseline + Matrix Factorization
2. Two-Tower retrieval model (FAISS exact inner-product index)
3. Neural Collaborative Filtering (NeuMF)
4. Deep & Cross ranking model (DCN-v2)
5. Negative sampling: uniform vs in-batch vs in-batch + mined hard negatives
6. Cold-start users: warm vs cold HitRate@20, and a content-based fallback embedding
7. Popularity bias of the retrieval stage (Gini, top-10% exposure, coverage)
8. Diversity: MMR lambda sweep (NDCG@20 vs intra-list diversity)
9. Latency: FAISS retrieval vs DCN scoring (in-process) and end-to-end HTTP
"""

import time
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
from sklearn.metrics import roc_auc_score

from data_pipeline import SyntheticRecSysDataGenerator, MerlinFeaturePipeline
from models import MatrixFactorization, TwoTowerModel, NeuralCollaborativeFiltering, DeepRankingModel

SEED = 42
EMB_DIM = 32


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def iterate_minibatches(tensors: dict, batch_size: int, shuffle: bool = True):
    n = len(tensors['user_id'])
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        idx = order[start:start + batch_size]
        yield {k: v[idx] for k, v in tensors.items()}


class InvestigationEngine:
    def __init__(self, num_users: int = 3000, num_items: int = 5000, num_interactions: int = 150000):
        set_seed()
        print("Initializing synthetic data and feature pipeline...")
        self.generator = SyntheticRecSysDataGenerator(num_users, num_items, num_interactions, seed=SEED)
        self.events_df, self.users_df, self.items_df = self.generator.generate()
        self.cold_events_df = self.generator.cold_events_df
        self.cold_user_ids = self.generator.cold_user_ids
        self.num_users = len(self.users_df)
        self.num_items = len(self.items_df)

        # Temporal split: events are sorted by timestamp, so train on the first 80%
        # and test on the most recent 20% (no future information in training).
        n = len(self.events_df)
        v0, t0 = int(n * 0.72), int(n * 0.8)
        self.train_df = self.events_df.iloc[:v0].reset_index(drop=True)
        self.val_df = self.events_df.iloc[v0:t0].reset_index(drop=True)   # model selection only
        self.test_df = self.events_df.iloc[t0:].reset_index(drop=True)    # reported numbers only

        # Feature statistics are fitted on TRAIN only.
        self.pipeline = MerlinFeaturePipeline(embedding_dim=EMB_DIM).fit(self.train_df, self.users_df, self.items_df)
        self.train_t = self.pipeline.transform(self.train_df)
        self.test_t = self.pipeline.transform(self.test_df)

        # Positive (label=1) interactions are what retrieval should surface.
        self.train_pos_df = self.train_df[self.train_df.label == 1].reset_index(drop=True)
        self.train_pos_t = self.pipeline.transform(self.train_pos_df)
        self.train_seen = self.train_pos_df.groupby('user_id')['item_id'].apply(set).to_dict()
        # Retrieval target: positive items a user engages with in the test period that they
        # had NOT already engaged with in training (re-recommending seen items is trivial).
        def new_positives(df):
            pos = df[df.label == 1].groupby('user_id')['item_id'].apply(set).to_dict()
            pos = {u: items - self.train_seen.get(u, set()) for u, items in pos.items()}
            return {u: items for u, items in pos.items() if items}
        self.test_pos_all = self.test_df[self.test_df.label == 1].groupby('user_id')['item_id'].apply(set).to_dict()
        self.test_pos = new_positives(self.test_df)
        self.val_pos = new_positives(self.val_df)
        # Cold users are evaluated on the SAME time window as warm users (the test period).
        t_test = self.test_df['timestamp'].min()
        cold_test = self.cold_events_df[(self.cold_events_df.timestamp >= t_test) & (self.cold_events_df.label == 1)]
        self.cold_pos = cold_test.groupby('user_id')['item_id'].apply(set).to_dict()
        self.all_items_t = self.pipeline.item_feature_batch(np.arange(self.num_items))

        # Retrieval evaluation users (fixed sample for speed and reproducibility).
        rng = np.random.RandomState(SEED)
        eval_users = [u for u in self.test_pos if u in self.train_seen]
        self.eval_users = rng.choice(eval_users, size=min(800, len(eval_users)), replace=False)
        val_users = [u for u in self.val_pos if u in self.train_seen]
        self.val_users = rng.choice(val_users, size=min(800, len(val_users)), replace=False)

        print(f"  {len(self.events_df)} warm events | train {len(self.train_df)} / val {len(self.val_df)} / test {len(self.test_df)} "
              f"| {len(self.cold_events_df)} cold-user events | positive rate {self.events_df.label.mean():.3f}")

    # ------------------------------------------------------------------ training
    def _train_pointwise(self, model, forward, epochs: int, lr: float, batch_size: int = 1024, weight_decay: float = 1e-6):
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        for _ in range(epochs):
            model.train()
            for mb in iterate_minibatches(self.train_t, batch_size):
                opt.zero_grad()
                loss = criterion(forward(model, mb), mb['label'])
                loss.backward()
                opt.step()
        model.eval()
        return model

    def _tt_forward(self, model, mb):
        u = model.user_tower(mb['user_id'], mb['gender'], mb['age'], mb['log_user_count'])
        i = model.item_tower(mb['item_id'], mb['category_id'], mb['price'], mb['quality_score'], mb['log_item_count'])
        return u, i

    def _item_embeddings(self, model) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            t = self.all_items_t
            return model.item_tower(t['item_id'], t['category_id'], t['price'], t['quality_score'], t['log_item_count']).numpy()

    def _user_embeddings(self, model, user_ids, fallback: str = None) -> np.ndarray:
        """fallback=None: use the user's own ID embedding.
        fallback='content': use the reserved UNKNOWN-user embedding (learned via ID dropout), so the
        vector is driven by content features only (gender, age)."""
        model.eval()
        with torch.no_grad():
            t = self.pipeline.user_feature_batch(user_ids)
            tower = model.user_tower
            u_id = tower.user_embed(t['user_id'])
            if fallback == 'content':
                u_id = tower.user_embed(torch.full_like(t['user_id'], self.num_users))
            g = tower.gender_embed(t['gender'])
            cont = torch.stack([t['age'], t['log_user_count']], dim=1)
            out = tower.mlp(torch.cat([u_id, g, cont], dim=1))
            return F.normalize(out, p=2, dim=1).numpy()

    def train_two_tower(self, negative_strategy: str = 'in_batch', epochs: int = 12, batch_size: int = 512,
                        lr: float = 0.005, n_uniform: int = 127, track_recall: bool = False,
                        id_dropout: float = 0.0):
        """negative_strategy:
          'uniform'   - each positive is contrasted with n_uniform items drawn uniformly from the catalog
          'in_batch'  - the other items in the batch are the negatives (popularity-skewed, free),
                        with logQ correction
          'hard'      - in-batch negatives + one mined hard negative per user (sampled from the user's
                        current top-50 scored items, excluding the positive)"""
        set_seed()
        model = TwoTowerModel(self.num_users, self.num_items, embedding_dim=EMB_DIM)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        # logQ correction (Yi et al., 2019): in-batch negatives are sampled in proportion to
        # popularity, so subtract log q(item) from each logit to avoid over-penalising popular items.
        counts = np.bincount(self.train_pos_df['item_id'].values, minlength=self.num_items) + 1.0
        log_q = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32)
        recall_curve = []
        for epoch in range(epochs):
            model.train()
            for mb in iterate_minibatches(self.train_pos_t, batch_size):
                if len(mb['user_id']) < 2:
                    continue
                if id_dropout > 0:
                    # ID dropout: with prob p, replace the user id by a reserved UNKNOWN id and zero the
                    # activity feature, so the UNKNOWN embedding learns a content-only (demographic) user.
                    drop = torch.rand(len(mb['user_id'])) < id_dropout
                    mb = dict(mb)
                    mb['user_id'] = torch.where(drop, torch.full_like(mb['user_id'], self.num_users), mb['user_id'])
                    mb['log_user_count'] = torch.where(drop, torch.zeros_like(mb['log_user_count']), mb['log_user_count'])
                opt.zero_grad()
                u, i = self._tt_forward(model, mb)
                tau = model.temperature
                if negative_strategy == 'uniform':
                    # Encode positives and uniform negatives in ONE item-tower pass so BatchNorm
                    # statistics are shared (separate passes let the model tell them apart by batch stats).
                    neg_ids = torch.tensor(np.random.randint(0, self.num_items, size=n_uniform))
                    ids = torch.cat([mb['item_id'], neg_ids])
                    at = {k: v[ids] for k, v in self.all_items_t.items()}
                    at['log_item_count'] = torch.cat([mb['log_item_count'], self.all_items_t['log_item_count'][neg_ids]])
                    u = model.user_tower(mb['user_id'], mb['gender'], mb['age'], mb['log_user_count'])
                    allv = model.item_tower(at['item_id'], at['category_id'], at['price'], at['quality_score'], at['log_item_count'])
                    i, neg = allv[:len(u)], allv[len(u):]
                    pos_logit = (u * i).sum(1, keepdim=True) / tau
                    neg_logit = u @ neg.T / tau
                    logits = torch.cat([pos_logit, neg_logit], dim=1)
                    loss = F.cross_entropy(logits, torch.zeros(len(u), dtype=torch.long))
                else:
                    logits = u @ i.T / tau - log_q[mb['item_id']].unsqueeze(0)
                    if negative_strategy == 'hard':
                        with torch.no_grad():
                            all_items = self._item_embeddings(model)
                            scores = u.detach().numpy() @ all_items.T
                            top = np.argsort(-scores, axis=1)[:, :50]
                            pos_items = mb['item_id'].numpy()
                            hard_ids = []
                            for r in range(len(top)):
                                cand = top[r][top[r] != pos_items[r]]
                                hard_ids.append(np.random.choice(cand))
                        model.train()
                        hard_ids = torch.tensor(hard_ids)
                        ids = torch.cat([mb['item_id'], hard_ids])
                        at = {k: v[ids] for k, v in self.all_items_t.items()}
                        u = model.user_tower(mb['user_id'], mb['gender'], mb['age'], mb['log_user_count'])
                        allv = model.item_tower(at['item_id'], at['category_id'], at['price'], at['quality_score'], at['log_item_count'])
                        i, h = allv[:len(u)], allv[len(u):]
                        logits = u @ i.T / tau - log_q[mb['item_id']].unsqueeze(0)
                        hard_logit = (u * h).sum(1, keepdim=True) / tau - log_q[hard_ids].unsqueeze(1)
                        logits = torch.cat([logits, hard_logit], dim=1)
                    loss = F.cross_entropy(logits, torch.arange(len(u)))
                loss.backward()
                opt.step()
            if track_recall:
                r = self.retrieval_metrics(model, k_recall=50, split='val')['Recall@50']
                recall_curve.append(r)
                if r >= max(recall_curve):
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if track_recall:
            model.load_state_dict(best_state)   # keep the best-validation epoch
        model.eval()
        return model, recall_curve

    def train_models(self, epochs: int = 8):
        print("\nTraining models...")
        set_seed()
        print("  - Matrix Factorization")
        self.mf_model = self._train_pointwise(MatrixFactorization(self.num_users + 1, self.num_items + 1, EMB_DIM),
                                              lambda m, b: m(b['user_id'], b['item_id']), epochs, lr=0.005)
        print("  - Neural Collaborative Filtering")
        self.ncf_model = self._train_pointwise(NeuralCollaborativeFiltering(self.num_users, self.num_items),
                                               lambda m, b: m(b['user_id'], b['item_id']), epochs, lr=0.002)
        print("  - DCN-v2 ranking model")
        self.rank_model = self._train_pointwise(DeepRankingModel(self.num_users, self.num_items, EMB_DIM),
                                                lambda m, b: m(b), epochs, lr=0.002)
        print("  - Two-Tower retrieval model (in-batch softmax + logQ, ID dropout 0.1)")
        self.tt_model, _ = self.train_two_tower('in_batch', epochs=12, id_dropout=0.1, track_recall=True)
        self.item_embs = self._item_embeddings(self.tt_model)
        self.faiss_index = faiss.IndexFlatIP(EMB_DIM)  # exact inner-product search (= cosine on unit vectors)
        self.faiss_index.add(self.item_embs)
        print("Training complete.")

    # ---------------------------------------------------------------- evaluation
    def _ranking_metrics(self, user_vecs, users, positives, k_hit=20, k_recall=50, exclude_seen=True):
        _, I = self.faiss_index_for(user_vecs).search(user_vecs, k_recall + 200)
        hits, recalls, ndcgs = [], [], []
        for row, u in enumerate(users):
            seen = self.train_seen.get(u, set()) if exclude_seen else set()
            ranked = [i for i in I[row] if i not in seen][:k_recall]
            pos = positives[u]
            top_hit = ranked[:k_hit]
            hits.append(float(any(i in pos for i in top_hit)))
            recalls.append(len(pos.intersection(ranked)) / len(pos))
            dcg = sum(1.0 / np.log2(r + 2) for r, i in enumerate(top_hit) if i in pos)
            idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(pos), k_hit)))
            ndcgs.append(dcg / idcg)
        return {f'HitRate@{k_hit}': float(np.mean(hits)), f'Recall@{k_recall}': float(np.mean(recalls)),
                f'NDCG@{k_hit}': float(np.mean(ndcgs))}

    def faiss_index_for(self, _):
        return self._active_index

    def retrieval_metrics(self, model, k_recall=50, split='val'):
        item_embs = self._item_embeddings(model)
        self._active_index = faiss.IndexFlatIP(EMB_DIM)
        self._active_index.add(item_embs)
        users, pos = (self.val_users, self.val_pos) if split == 'val' else (self.eval_users, self.test_pos)
        uv = self._user_embeddings(model, users)
        return self._ranking_metrics(uv, users, pos, k_recall=k_recall)

    def _tt_auc(self, model):
        with torch.no_grad():
            u, i = self._tt_forward(model, self.test_t)
            scores = (u * i).sum(1).numpy()
        return float(roc_auc_score(self.test_t['label'].numpy(), scores))

    def evaluate_auc(self):
        labels = self.test_t['label'].numpy()
        with torch.no_grad():
            pop_scores = self.test_t['log_item_count'].numpy()
            mf = self.mf_model(self.test_t['user_id'], self.test_t['item_id']).numpy()
            ncf = self.ncf_model(self.test_t['user_id'], self.test_t['item_id']).numpy()
            dcn = self.rank_model(self.test_t).numpy()
        return {
            'Popularity Baseline': float(roc_auc_score(labels, pop_scores)),
            'Matrix Factorization': float(roc_auc_score(labels, mf)),
            'Two-Tower Model': self._tt_auc(self.tt_model),
            'Neural Collaborative Filtering': float(roc_auc_score(labels, ncf)),
            'Deep Ranking Model': float(roc_auc_score(labels, dcn)),
        }

    def evaluate_negative_sampling(self, epochs: int = 12):
        out = {}
        names = {'uniform': 'Uniform Random Negatives', 'in_batch': 'In-Batch Softmax Negatives',
                 'hard': 'In-Batch + Mined Hard Negatives'}
        for strat, name in names.items():
            print(f"  - negative sampling: {name}")
            model, curve = self.train_two_tower(strat, epochs=epochs, track_recall=True)
            best = max(curve)
            converged = next(e + 1 for e, r in enumerate(curve) if r >= 0.98 * best)
            test_recall = self.retrieval_metrics(model, k_recall=50, split='test')['Recall@50']
            out[name] = {'AUC': round(self._tt_auc(model), 4), 'Recall@50': round(test_recall, 4),
                         'Convergence Epochs': int(converged),
                         'Val Recall@50 by epoch': [round(r, 4) for r in curve]}
        return out

    def evaluate_cold_start(self):
        """Like-for-like comparison: for every group the target is the user's positive items in the
        evaluation period, and no items are filtered out (cold users have no history to filter)."""
        self._active_index = self.faiss_index
        cold_users = np.array(sorted(self.cold_pos.keys()))
        warm_users = np.array([u for u in self.eval_users if u in self.test_pos_all])
        m = lambda uv, users, pos: self._ranking_metrics(uv, users, pos, exclude_seen=False)
        warm = m(self._user_embeddings(self.tt_model, warm_users), warm_users, self.test_pos_all)
        cold = m(self._user_embeddings(self.tt_model, cold_users), cold_users, self.cold_pos)
        fb = m(self._user_embeddings(self.tt_model, cold_users, fallback='content'), cold_users, self.cold_pos)
        top_pop = [i for i, _ in sorted(self.pipeline.pop_counts.items(), key=lambda x: -x[1])][:20]
        pop_hr = float(np.mean([any(i in self.cold_pos[u] for i in top_pop) for u in cold_users]))
        drop = 1 - cold['HitRate@20'] / warm['HitRate@20']
        gap = warm['HitRate@20'] - cold['HitRate@20']
        recovered = (fb['HitRate@20'] - cold['HitRate@20']) / gap if gap > 0 else float('nan')
        r = lambda x: {'HitRate@20': round(x['HitRate@20'], 4), 'NDCG@20': round(x['NDCG@20'], 4)}
        return {
            'Warm Users': r(warm),
            'Cold Users (untrained ID embedding)': r(cold),
            'Cold Users + Content Fallback (UNKNOWN-id embedding)': r(fb),
            'Cold Users + Popularity Fallback': {'HitRate@20': round(pop_hr, 4)},
            'summary': {'cold_drop_pct': round(100 * drop, 1), 'gap_recovered_pct': round(100 * recovered, 1),
                        'n_cold_users': int(len(cold_users)), 'n_warm_users': int(len(warm_users))},
        }

    def _gini_coefficient(self, array):
        array = np.sort(array)
        index = np.arange(1, array.shape[0] + 1)
        n = array.shape[0]
        return float(np.sum((2 * index - n - 1) * array) / (n * np.sum(array)))

    def evaluate_popularity_bias(self, k=20):
        _, I = self.faiss_index.search(self._user_embeddings(self.tt_model, self.eval_users), k)
        counts = np.bincount(I.ravel(), minlength=self.num_items)
        recommended = counts[counts > 0]
        top10 = np.sort(counts)[::-1][:int(self.num_items * 0.1)].sum() / counts.sum()
        return {'Gini Concentration Coefficient': round(self._gini_coefficient(counts), 3),
                'Top 10% Item Exposure Share': round(float(top10), 3),
                'Catalog Coverage Rate': round(len(recommended) / self.num_items, 3)}

    # ------------------------------------------------------------- two-stage + MMR
    def score_candidates(self, user_id: int, item_ids: np.ndarray) -> np.ndarray:
        n = len(item_ids)
        batch = {k: v.repeat(n) for k, v in self.pipeline.user_feature_batch([user_id]).items()}
        batch.update(self.pipeline.item_feature_batch(item_ids))
        with torch.no_grad():
            return torch.sigmoid(self.rank_model(batch)).numpy()

    def mmr_rerank(self, item_ids, scores, top_k=20, lam=0.8):
        """Maximal Marginal Relevance: argmax  lam*rel(i) - (1-lam)*max_{j in S} sim(i,j)."""
        embs = self.item_embs[item_ids]
        rel = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        selected, remaining = [], list(range(len(item_ids)))
        while remaining and len(selected) < top_k:
            if selected:
                sim = embs[remaining] @ embs[selected].T
                mmr = lam * rel[remaining] - (1 - lam) * sim.max(axis=1)
            else:
                mmr = lam * rel[remaining]
            best = remaining[int(np.argmax(mmr))]
            selected.append(best)
            remaining.remove(best)
        return [item_ids[s] for s in selected]

    def recommend(self, user_id: int, top_k: int = 20, n_candidates: int = 100, lam: float = 1.0):
        uv = self._user_embeddings(self.tt_model, [user_id])
        D, I = self.faiss_index.search(uv, n_candidates + 200)
        seen = self.train_seen.get(user_id, set())
        keep = [j for j, i in enumerate(I[0]) if i not in seen][:n_candidates]
        cands, sims = I[0][keep], D[0][keep]
        # Final score ~ log P(engage) = log P(exposure | user) + log P(convert | exposed):
        # stage 1 supplies the first term (similarity / temperature), DCN-v2 the second.
        p_convert = self.score_candidates(user_id, cands)
        scores = sims / self.tt_model.temperature + np.log(p_convert + 1e-9)
        if lam >= 1.0:
            order = np.argsort(-scores)[:top_k]
            return cands[order], scores[order]
        picked = self.mmr_rerank(cands, scores, top_k=top_k, lam=lam)
        score_map = dict(zip(cands, scores))
        return np.array(picked), np.array([score_map[i] for i in picked])

    def evaluate_diversity(self, lambdas=(1.0, 0.8, 0.5, 0.2, 0.0), n_users=300):
        users = self.eval_users[:n_users]
        out = []
        for lam in lambdas:
            ndcgs, ilds = [], []
            for u in users:
                recs, _ = self.recommend(int(u), top_k=20, lam=lam)
                pos = self.test_pos[u]
                dcg = sum(1.0 / np.log2(r + 2) for r, i in enumerate(recs) if i in pos)
                idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(pos), 20)))
                ndcgs.append(dcg / idcg)
                e = self.item_embs[recs]
                sim = e @ e.T
                ilds.append(1 - (sim.sum() - np.trace(sim)) / (len(recs) * (len(recs) - 1)))
            out.append({'lambda_mmr': lam, 'Relevance NDCG@20': round(float(np.mean(ndcgs)), 4),
                        'Intra-List Diversity (ILD)': round(float(np.mean(ilds)), 4)})
        return out

    def benchmark_latency(self, runs: int = 200):
        user = int(self.eval_users[0])
        uv = self._user_embeddings(self.tt_model, [user])
        retrieval, ranking = [], []
        _, I = self.faiss_index.search(uv, 100)
        cands = I[0]
        batch = {k: v.repeat(100) for k, v in self.pipeline.user_feature_batch([user]).items()}
        batch.update(self.pipeline.item_feature_batch(cands))
        for _ in range(runs):
            t0 = time.perf_counter()
            self.faiss_index.search(uv, 100)
            retrieval.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            with torch.no_grad():
                self.rank_model(batch)
            ranking.append((time.perf_counter() - t0) * 1000)
        pct = lambda a: {f'p{p}_ms': round(float(np.percentile(a, p)), 3) for p in (50, 95, 99)}
        return {'Candidate Retrieval (FAISS exact IP - 5k items, top-100)': pct(retrieval),
                'Deep Scoring Model (DCN - 100 candidates)': pct(ranking)}

    def evaluate_all(self):
        results = {'dataset': {'warm_events': int(len(self.events_df)), 'train_events': int(len(self.train_df)),
                               'val_events': int(len(self.val_df)), 'test_events': int(len(self.test_df)), 'cold_user_events': int(len(self.cold_events_df)),
                               'positive_rate': round(float(self.events_df.label.mean()), 3),
                               'split': 'temporal 72/8/20 (train/val/test)'}}
        print("\nEvaluating...")
        results['models_auc'] = {k: round(v, 4) for k, v in self.evaluate_auc().items()}
        self._active_index = self.faiss_index
        results['two_tower_retrieval'] = {k: round(v, 4) for k, v in self._ranking_metrics(
            self._user_embeddings(self.tt_model, self.eval_users), self.eval_users, self.test_pos).items()}
        results['negative_sampling'] = self.evaluate_negative_sampling()
        results['cold_start'] = self.evaluate_cold_start()
        results['popularity_bias'] = self.evaluate_popularity_bias()
        results['diversity'] = self.evaluate_diversity()
        results['latency'] = self.benchmark_latency()
        return results


if __name__ == '__main__':
    engine = InvestigationEngine()
    engine.train_models()
    results = engine.evaluate_all()
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nINVESTIGATION RESULTS SUMMARY:")
    print(json.dumps(results, indent=2))
