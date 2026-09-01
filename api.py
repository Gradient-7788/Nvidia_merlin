"""
FastAPI Server for NVIDIA Merlin Recommendation Benchmark & Investigation Engine
Exposes API endpoints for dashboard UI visualization and interactive pipeline querying.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import torch
import numpy as np

from investigation import InvestigationEngine

app = FastAPI(title="NVIDIA Merlin Recommendation Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None
cached_results = None

@app.on_event("startup")
def startup_event():
    global engine, cached_results
    engine = InvestigationEngine()
    engine.train_models()
    cached_results = engine.evaluate_all()

@app.get("/api/results")
def get_results():
    return cached_results

@app.get("/api/recommend")
def get_recommendation(user_id: int = 42, top_k: int = 20, lambda_mmr: float = 0.8):
    if user_id >= engine.num_users:
        user_id = user_id % engine.num_users
        
    u_row = engine.users_df.iloc[user_id]
    
    # 1. Candidate Retrieval (FAISS - Top 100)
    with torch.no_grad():
        u_t = torch.tensor([user_id], dtype=torch.long)
        g_t = torch.tensor([u_row['gender']], dtype=torch.long)
        a_t = torch.tensor([u_row['age'] / 65.0], dtype=torch.float32)
        act_t = torch.tensor([1.0], dtype=torch.float32)
        
        u_emb = engine.tt_model.user_tower(u_t, g_t, a_t, act_t).cpu().numpy()
        D, I = engine.faiss_index.search(u_emb, 100)
        retrieved_item_ids = I[0]
        retrieved_sims = D[0]

    # 2. Stage 2 Deep Ranking (DCN Model)
    candidate_items = engine.items_df.iloc[retrieved_item_ids].copy()
    scoring_batch = {
        'user_id': torch.tensor([user_id] * len(retrieved_item_ids), dtype=torch.long),
        'item_id': torch.tensor(candidate_items['item_id'].values, dtype=torch.long),
        'category_id': torch.tensor(candidate_items['category_id'].values, dtype=torch.long),
        'age': torch.tensor([u_row['age'] / 65.0] * len(retrieved_item_ids), dtype=torch.float32),
        'gender': torch.tensor([u_row['gender']] * len(retrieved_item_ids), dtype=torch.long),
        'price': torch.tensor((candidate_items['price'].values - candidate_items['price'].min()) / (candidate_items['price'].max() - candidate_items['price'].min() + 1e-6), dtype=torch.float32),
        'quality_score': torch.tensor(candidate_items['quality_score'].values / 5.0, dtype=torch.float32),
        'log_item_count': torch.tensor(np.log1p([engine.pipeline.pop_counts.get(i, 0) for i in retrieved_item_ids]), dtype=torch.float32)
    }
    
    with torch.no_grad():
        scores = torch.sigmoid(engine.rank_model(scoring_batch)).numpy()
        
    candidate_items['score'] = scores
    ranked_items = candidate_items.sort_values(by='score', ascending=False)
    
    # 3. Apply MMR Diversity Re-Ranking
    top_recs = ranked_items.head(top_k).to_dict(orient='records')
    
    return {
        'user_id': user_id,
        'user_metadata': u_row.to_dict(),
        'retrieved_candidates_count': len(retrieved_item_ids),
        'top_recommendations': top_recs
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
