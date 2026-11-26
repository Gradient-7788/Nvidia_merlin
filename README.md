# NVIDIA Merlin — Large-Scale Two-Stage Recommendation System

> An end-to-end, research-grade recommendation engine inspired by [NVIDIA Merlin](https://github.com/NVIDIA-Merlin/Merlin), implementing a complete **two-stage retrieval + ranking pipeline** with interactive benchmarking dashboard and empirical investigation suite.

---

## System Architecture

```
User Events (synthetic latent-taste simulator, 150k generated)
     │
     ▼
┌─────────────────────────────────┐
│   Feature Pipeline (NVTabular-  │  train-only statistics: MinMax,
│   style, data_pipeline.py)      │  log counts, categorical IDs
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│   Candidate Generation          │  Two-Tower Dual-Encoder + FAISS
│   (Stage 1 Retrieval)           │  exact inner-product index
└─────────────────────────────────┘
     │  top-100 candidates (< 0.1 ms)
     ▼
┌─────────────────────────────────┐
│   Ranking Model (Stage 2)       │  Deep & Cross Network (DCN-v2)
│   Fine-scores all candidates    │  + Deep MLP fusion head
└─────────────────────────────────┘
     │
     ▼
   Top 20 Recommendations
   (optional MMR diversity re-ranking)
```

---

## Project Structure

```
Nvidia_merlin/
├── data_pipeline.py       # Synthetic data generator + NVTabular-style feature pipeline
├── models.py              # PyTorch model architectures (MF, Two-Tower, NCF, DCN)
├── investigation.py       # Comprehensive benchmarking engine (9 investigation topics)
├── api.py                 # FastAPI REST backend serving all endpoints
├── dashboard/             # React + Vite + Tailwind interactive web UI
│   ├── src/
│   │   ├── App.jsx        # Main dashboard with 3 interactive tabs
│   │   ├── index.css      # NVIDIA-themed glassmorphism dark styling
│   │   └── main.jsx       # React entry point
│   ├── index.html         # SEO-optimized HTML with Inter font
│   ├── tailwind.config.js # Tailwind v3 configuration
│   ├── postcss.config.js  # PostCSS plugin wiring
│   ├── vite.config.js     # Vite dev server config
│   └── package.json       # Node dependencies
└── README.md              # This file
```

---

## Models Implemented

### 1. Matrix Factorization Baseline
Classic collaborative filtering with learned user/item embeddings and bias terms. Serves as the baseline for all comparisons.

### 2. Two-Tower Dual-Encoder (Candidate Retrieval)
- **User Tower**: User ID embedding + gender embedding + continuous features (age, activity) → MLP → L2-normalized vector
- **Item Tower**: Item ID embedding + category embedding + continuous features (price, quality, popularity) → MLP → L2-normalized vector
- **Training**: In-batch softmax loss with temperature scaling (τ = 0.07) and logQ popularity correction; ID dropout (p = 0.1) into a reserved UNKNOWN-user embedding for cold start
- **Inference**: All item embeddings indexed in **FAISS** (`IndexFlatIP`, exact inner-product search = cosine on unit vectors)

### 3. Neural Collaborative Filtering (NeuMF)
Fuses two pathways:
- **GMF path**: Element-wise product of user/item embeddings (linear interaction)
- **MLP path**: Concatenated user/item embeddings through a deep 3-layer MLP (non-linear interaction)
- Final fusion layer combines both for binary prediction

### 4. Deep & Cross Ranking Model (DCN-v2)
Stage 2 scoring model with:
- **Cross Network**: Explicit bounded-degree feature interactions (2 cross layers)
- **Deep Network**: 3-layer MLP (128 → 64 → 32) with BatchNorm
- Parallel cross + deep outputs concatenated → linear prediction head

---

## Investigation Topics

The investigation engine (`investigation.py`) empirically evaluates **9 key topics**:

| # | Topic | Key Metrics |
|---|-------|-------------|
| 1 | **Matrix Factorization Baseline** | AUC, convergence speed |
| 2 | **Two-Tower Retrieval Model** | In-batch softmax loss, FAISS recall |
| 3 | **Neural Collaborative Filtering** | AUC comparison vs MF |
| 4 | **Deep Ranking Model (DCN)** | AUC, cross-feature interaction quality |
| 5 | **Negative Sampling** | Uniform vs In-Batch vs In-Batch + mined hard negatives (AUC, Recall@50, convergence epochs on validation) |
| 6 | **Cold-Start** | Warm vs cold users (same test window), content-based fallback vs popularity fallback |
| 7 | **Popularity Bias** | Gini concentration coefficient, top-10% exposure share, catalog coverage rate |
| 8 | **Diversity** | MMR λ trade-off curve: Relevance NDCG@20 vs Intra-List Diversity (ILD) |
| 9 | **Latency** | Candidate retrieval p50/p95/p99 (ms) vs Deep ranking scoring latency |

---

## Dashboard

The interactive React dashboard provides three views:

| Tab | Description |
|-----|-------------|
| **Architecture Pipeline** | Visual 5-stage pipeline diagram, latency KPI cards, architecture explanations |
| **Models & Benchmarks** | AUC bar chart comparison, negative sampling table, cold-start degradation analysis, MMR diversity-relevance tradeoff line chart, latency percentile breakdown |
| **Live Rec Simulator** | Select any user ID, adjust MMR diversity weight (λ), run the full two-stage pipeline in real-time, and see scored Top-20 recommendations |

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- pip packages: `torch`, `numpy`, `pandas`, `scikit-learn`, `faiss-cpu`, `scipy`, `fastapi`, `uvicorn`

### 1. Install Python Dependencies

```bash
pip install torch numpy pandas scikit-learn faiss-cpu scipy fastapi uvicorn
```

### 2. Run the Investigation Suite (Standalone)

```bash
cd Nvidia_merlin
python3 investigation.py
```

This will:
- Generate 150k synthetic user events across 3,000 users and 5,000 items (latent-taste simulator)
- Train all 4 models (MF, Two-Tower, NCF, DCN)
- Build FAISS vector index
- Run all 9 investigation benchmarks
- Write every metric to `results.json` and print it (nothing is hardcoded)

### 3. Start the API Server

```bash
python3 api.py
```

The FastAPI server starts on **http://localhost:8000** with endpoints:
- `GET /api/results` — Full benchmark results JSON
- `GET /api/recommend?user_id=42&top_k=20&lambda_mmr=0.8` — Live two-stage recommendations

### 4. Start the Dashboard

```bash
cd dashboard
npm install
npm run dev -- --port 3000
```

Open **http://localhost:3000** in your browser.

---

## API Reference

### `GET /api/results`

Returns the full investigation benchmarks including model AUCs, negative sampling comparisons, cold-start metrics, popularity bias stats, diversity tradeoff curves, and latency percentiles.

### `GET /api/recommend`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_id` | int | 42 | Target user ID for recommendations |
| `top_k` | int | 20 | Number of top recommendations to return |
| `lambda_mmr` | float | 0.8 | MMR diversity weight (1.0 = pure relevance, 0.0 = pure diversity) |

**Response**: User metadata, retrieved candidate count, and scored top-K item recommendations.

---

## Key Design Decisions

1. **Latent-taste synthetic data.** Users have a 10-dim category-taste vector, partly explained by their demographic segment (age bucket × gender). Exposure probability ∝ popularity^0.7 × exp(1.5 · taste[category]); the binary label (e.g. purchase after view) depends on taste, quality and price plus noise. Item popularity follows a power law (α = 1.2). An oracle that knows the true signal reaches AUC ≈ 0.93, so there is real structure to learn. The last 5% of user IDs and items never appear in training (cold-start).
2. **Temporal 72 / 8 / 20 split.** Train on the oldest events, select epochs on validation, report on the most recent 20%. Feature statistics (popularity, activity, scaling bounds) are fitted on train only.
3. **In-batch softmax with logQ correction.** In-batch negatives are sampled in proportion to popularity; subtracting log q(item) from each logit (Yi et al., 2019) stops popular items from being over-penalised.
4. **FAISS `IndexFlatIP`.** Exact search: at 5,000 × 32 dims brute force is sub-0.1 ms with perfect recall. `IndexIVFFlat` / `IndexHNSW` are the swap at larger catalogs.
5. **DCN-v2 cross network.** Full-rank cross layers x₍ₗ₊₁₎ = x₀ ⊙ (Wₗxₗ + bₗ) + xₗ; two layers give explicit feature interactions up to degree 3, in parallel with a deep MLP.
6. **Two-stage scoring.** Final score ≈ log P(engage) = similarity/τ (stage 1, exposure) + log P(convert | exposed) (stage 2, DCN-v2).
7. **Cold-start via ID dropout.** During two-tower training 10% of user IDs are replaced by a reserved UNKNOWN id, so that embedding learns a content-only (demographic) user representation that new users fall back to.
8. **MMR re-ranking.** argmax λ·rel(i) − (1−λ)·maxⱼ∈S sim(i, j), with similarity from item-tower embeddings.

---

## Results (from `results.json`, seed 42)

Dataset: 132,977 warm events (train 95,743 / val 10,638 / test 26,596), 7,569 cold-user events, positive rate 0.833.

```
Ranking AUC on the test period (pointwise labels):
  Popularity Baseline:             0.525
  Two-Tower (dot product):         0.717
  Neural Collaborative Filtering:  0.810
  DCN-v2 Ranking Model:            0.833
  Matrix Factorization:            0.843

Two-tower retrieval (new positives, seen items excluded):
  HitRate@20 0.652 | Recall@50 0.304 | NDCG@20 0.160

Negative sampling (test Recall@50 at best-validation epoch; epochs to reach 98% of best val Recall@50):
  Uniform Random Negatives           Recall@50 0.298 | AUC 0.686 | 7 epochs
  In-Batch Softmax Negatives         Recall@50 0.299 | AUC 0.724 | 5 epochs
  In-Batch + Mined Hard Negatives    Recall@50 0.298 | AUC 0.712 | 4 epochs

Cold-start users (same test window, no filtering):
  Warm Users                                           HitRate@20 0.775
  Cold Users (untrained ID embedding)                  HitRate@20 0.581
  Cold Users + Content Fallback (UNKNOWN-id embedding) HitRate@20 0.730
  Cold Users + Popularity Fallback                     HitRate@20 0.696
  -> cold users lose 25.0% HitRate@20; the content fallback recovers 76.7% of that gap

Popularity bias of retrieval (top-20 for 800 users):
  Gini 0.98 | top-10% item exposure 1.0 | catalog coverage 0.058

Latency (in-process, CPU, 200 runs):
  FAISS retrieval (top-100 of 5k): p50 0.043 ms | p95 0.08 ms
  DCN-v2 scoring (100 candidates): p50 0.197 ms | p95 0.265 ms
  End-to-end GET /api/recommend (FastAPI TestClient, incl. HTTP + JSON): p50 ≈ 4.9 ms | p95 ≈ 8.1 ms
```

Honest notes: MF slightly beats DCN-v2 on AUC here because the simulator's signal is mostly a user × category interaction, which MF captures directly. Negative-sampling strategies end at similar Recall@50; the difference is convergence speed. Retrieval is strongly popularity-concentrated (low coverage) — MMR raises intra-list diversity at a small NDCG cost.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Data Pipeline | NumPy, Pandas (NVTabular-inspired ETL) |
| Models | PyTorch (MF, Two-Tower, NCF, DCN-v2) |
| Vector Index | FAISS (`faiss-cpu`) |
| Evaluation | scikit-learn (AUC), custom metrics |
| API Server | FastAPI + Uvicorn |
| Dashboard | React 18 + Vite 5 + Tailwind CSS 3 + Recharts |
| Fonts | Inter (Google Fonts) |

---

## References

- [NVIDIA Merlin](https://github.com/NVIDIA-Merlin/Merlin) — GPU-accelerated recommender framework
- [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular) — Feature engineering for tabular data
- [He et al., 2017](https://arxiv.org/abs/1708.05031) — Neural Collaborative Filtering (NeuMF)
- [Wang et al., 2021](https://arxiv.org/abs/2008.13535) — DCN V2: Improved Deep & Cross Network
- [Yi et al., 2019](https://dl.acm.org/doi/10.1145/3298689.3346996) — Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations
- [Carbonell & Goldstein, 1998](https://dl.acm.org/doi/10.1145/290941.291025) — Maximal Marginal Relevance (MMR)

---

## License

This project is for educational and research purposes. Inspired by the NVIDIA Merlin open-source ecosystem.
