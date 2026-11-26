"""
Merlin Recommendation Engine - Data & Feature Pipeline
Simulates scale User Events and NVTabular-style Feature Engineering.
"""

import numpy as np
import pandas as pd
import torch
import time
from typing import Dict, Tuple, List

class SyntheticRecSysDataGenerator:
    """Generates synthetic User Events with long-tail power law dynamics, cold-start entities, and rich metadata."""
    
    def __init__(self, num_users: int = 5000, num_items: int = 10000, num_interactions: int = 250000, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self.num_users = num_users
        self.num_items = num_items
        self.num_interactions = num_interactions
        
    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Latent-taste simulator.

        - Items: category (10), price, quality, and a power-law base popularity (alpha=1.2).
        - Users: age, gender, activity level, and a 10-dim category-taste vector. Taste is
          partly explained by the demographic segment (age bucket x gender) and partly
          individual, so demographics are a weak-but-real signal for cold-start users.
        - Exposure: user u sees item i with probability  ~ pop_i^0.7 * exp(1.5 * taste_u[cat_i]).
        - Label (e.g. purchase after view): 1 if  taste_u[cat_i] + 0.75*(quality_i-3)
          - 0.3*log(price_i/30) + N(0, 0.8)  >  0.
        """
        rng = np.random.RandomState(self.seed)
        n_cat = 10

        # ---- items
        item_categories = rng.choice(np.arange(n_cat), size=self.num_items,
                                     p=[0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.04, 0.04, 0.02])
        item_prices = rng.exponential(scale=30.0, size=self.num_items) + 4.99
        item_quality = rng.uniform(1.0, 5.0, size=self.num_items)
        alpha = 1.2
        pop_weight = np.arange(1, self.num_items + 1) ** (-alpha)
        rng.shuffle(pop_weight)  # popularity is not tied to item id order
        pop_weight /= pop_weight.sum()
        items_df = pd.DataFrame({'item_id': np.arange(self.num_items), 'category_id': item_categories,
                                 'price': item_prices, 'quality_score': item_quality, 'pop_weight': pop_weight})

        # ---- users
        user_age = rng.randint(18, 65, size=self.num_users)
        user_gender = rng.choice([0, 1], size=self.num_users)
        activity = rng.choice(['low', 'medium', 'high'], size=self.num_users, p=[0.5, 0.35, 0.15])
        age_bucket = np.digitize(user_age, [30, 42, 54])                  # 4 buckets
        segment_taste = rng.normal(0, 1.0, size=(4, 2, n_cat))            # demographic prior
        taste = segment_taste[age_bucket, user_gender] + rng.normal(0, 1.0, size=(self.num_users, n_cat))
        users_df = pd.DataFrame({'user_id': np.arange(self.num_users), 'age': user_age,
                                 'gender': user_gender, 'activity_level': activity})

        # ---- interactions: events per user proportional to activity level
        act_w = np.select([activity == 'low', activity == 'medium'], [1.0, 2.5], 6.0)
        n_events = np.maximum(1, np.round(act_w / act_w.sum() * self.num_interactions)).astype(int)
        log_pop = 0.7 * np.log(pop_weight)
        u_list, i_list = [], []
        for u in range(self.num_users):
            logits = log_pop + 1.5 * taste[u, item_categories]
            p = np.exp(logits - logits.max()); p /= p.sum()
            u_list.append(np.full(n_events[u], u)); i_list.append(rng.choice(self.num_items, size=n_events[u], p=p))
        user_ids = np.concatenate(u_list); item_ids = np.concatenate(i_list)

        signal = (taste[user_ids, item_categories[item_ids]]
                  + 0.75 * (item_quality[item_ids] - 3.0)
                  - 0.3 * np.log(item_prices[item_ids] / 30.0))
        events = (signal + rng.normal(0, 0.8, size=len(user_ids)) > 0).astype(int)
        timestamps = rng.randint(1600000000, 1700000000, size=len(user_ids))
        events_df = pd.DataFrame({'user_id': user_ids, 'item_id': item_ids, 'label': events,
                                  'timestamp': timestamps}).sort_values('timestamp').reset_index(drop=True)
        self.true_signal = None  # kept private; see oracle_auc in investigation for the ceiling

        # ---- cold-start entities: last 5% of user ids and items never appear in training.
        cold_item_ids = np.arange(int(self.num_items * 0.95), self.num_items)
        cold_user_ids = np.arange(int(self.num_users * 0.95), self.num_users)
        events_df = events_df[~events_df['item_id'].isin(cold_item_ids)]
        cold_mask = events_df['user_id'].isin(cold_user_ids)
        self.cold_events_df = events_df[cold_mask].reset_index(drop=True)
        self.cold_user_ids = cold_user_ids
        self.cold_item_ids = cold_item_ids
        self.taste = taste
        events_df = events_df[~cold_mask]
        return events_df.reset_index(drop=True), users_df, items_df

class MerlinFeaturePipeline:
    """NVTabular-inspired feature pipeline.

    Statistics (popularity counts, user activity counts, min/max for scaling) are
    fitted on the TRAINING events only and then applied to any split, so the
    test set never leaks into the features.
    """

    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim
        self.pop_counts = {}
        self.user_counts = {}

    def fit(self, train_events: pd.DataFrame, users_df: pd.DataFrame, items_df: pd.DataFrame):
        self.users_df = users_df
        self.items_df = items_df
        self.pop_counts = train_events['item_id'].value_counts().to_dict()
        self.user_counts = train_events['user_id'].value_counts().to_dict()
        self.price_min, self.price_max = items_df['price'].min(), items_df['price'].max()
        self.age_min, self.age_max = users_df['age'].min(), users_df['age'].max()
        return self

    # --- per-entity feature helpers (used for training, retrieval and serving) ---
    def norm_price(self, price):
        return (np.asarray(price) - self.price_min) / (self.price_max - self.price_min + 1e-6)

    def norm_age(self, age):
        return (np.asarray(age) - self.age_min) / (self.age_max - self.age_min + 1e-6)

    def log_item_count(self, item_ids):
        return np.log1p([self.pop_counts.get(int(i), 0) for i in item_ids])

    def log_user_count(self, user_ids):
        return np.log1p([self.user_counts.get(int(u), 0) for u in user_ids])

    def item_feature_batch(self, item_ids) -> Dict[str, torch.Tensor]:
        item_ids = np.asarray(item_ids)
        it = self.items_df.iloc[item_ids]
        return {
            'item_id': torch.tensor(item_ids, dtype=torch.long),
            'category_id': torch.tensor(it['category_id'].values, dtype=torch.long),
            'price': torch.tensor(self.norm_price(it['price'].values), dtype=torch.float32),
            'quality_score': torch.tensor(it['quality_score'].values / 5.0, dtype=torch.float32),
            'log_item_count': torch.tensor(self.log_item_count(item_ids), dtype=torch.float32),
        }

    def user_feature_batch(self, user_ids) -> Dict[str, torch.Tensor]:
        user_ids = np.asarray(user_ids)
        us = self.users_df.iloc[user_ids]
        return {
            'user_id': torch.tensor(user_ids, dtype=torch.long),
            'gender': torch.tensor(us['gender'].values, dtype=torch.long),
            'age': torch.tensor(self.norm_age(us['age'].values), dtype=torch.float32),
            'log_user_count': torch.tensor(self.log_user_count(user_ids), dtype=torch.float32),
        }

    def transform(self, events_df: pd.DataFrame) -> Dict[str, torch.Tensor]:
        tensors = {}
        tensors.update(self.user_feature_batch(events_df['user_id'].values))
        tensors.update(self.item_feature_batch(events_df['item_id'].values))
        tensors['label'] = torch.tensor(events_df['label'].values, dtype=torch.float32)
        return tensors

    def fit_transform(self, events_df: pd.DataFrame, users_df: pd.DataFrame, items_df: pd.DataFrame):
        """Backwards-compatible helper: fit on all given events and transform them."""
        self.fit(events_df, users_df, items_df)
        return self.transform(events_df), events_df

if __name__ == '__main__':
    generator = SyntheticRecSysDataGenerator()
    events_df, users_df, items_df = generator.generate()
    pipeline = MerlinFeaturePipeline()
    tensors, full_df = pipeline.fit_transform(events_df, users_df, items_df)
    print(f"Generated {len(events_df)} user events across {len(users_df)} users and {len(items_df)} items.")
    print("Tensor dictionary ready for Candidate Generation & Ranking stages!")
