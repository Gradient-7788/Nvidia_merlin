"""
FastAPI Server for NVIDIA Merlin Recommendation Benchmark & Investigation Engine
Exposes API endpoints for dashboard UI visualization and interactive pipeline querying.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    """Two-stage pipeline: FAISS retrieval (top-100, seen items removed) -> DCN-v2 scoring
    -> optional MMR diversity re-ranking (lambda_mmr=1.0 disables it)."""
    user_id = user_id % engine.num_users
    item_ids, scores = engine.recommend(user_id, top_k=top_k, n_candidates=100, lam=lambda_mmr)
    recs = engine.items_df.iloc[item_ids].copy()
    recs['score'] = scores
    return {
        'user_id': user_id,
        'user_metadata': engine.users_df.iloc[user_id].to_dict(),
        'is_cold_user': bool(user_id >= int(engine.num_users * 0.95)),
        'retrieved_candidates_count': 100,
        'lambda_mmr': lambda_mmr,
        'top_recommendations': recs.to_dict(orient='records'),
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
