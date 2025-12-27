# Personalized News Ranking Pipeline

## Quick Start

```bash
# 1. Start services
docker compose up -d

# 2. Index articles (once)
python indexing.py

# 3. Collect interactions
python baseline.py

# 4. Extract features & train all models
python scripts/extract_features.py
python scripts/train.py

# 5. Run A/B testing evaluation
python scripts/evaluate.py
```

## Available Models

| Model | Description | Strengths |
|-------|-------------|-----------|
| **GNN** | Graph Neural Network with attention | Captures user-article-topic relationships |
| **LTR** | Learning to Rank (GradientBoosting) | Feature-rich prediction |
| **CF** | Collaborative Filtering ensemble | User-based + Item-based + Matrix Factorization |
| **Content** | Content-based filtering | TF-IDF + topic similarity |
| **Popularity** | Popularity-based ranking | Cold-start fallback |
| **Ensemble** | Weighted combination of all models | Best overall performance |

## Pipeline Overview

```
baseline.py          →  interaction_logs.jsonl
                              ↓
extract_features.py  →  user_features.json
                         article_features.json
                         interaction_graph.pkl
                              ↓
train.py             →  gnn_model.pkl
                         ltr_model.pkl
                         cf_model.pkl
                         content_model.pkl
                         popularity_model.pkl
                         ensemble_model.pkl
                              ↓
evaluate.py          →  A/B test all strategies
                         Compare CTR/MRR/NDCG
                         artifacts/evaluation/ab_test_*.json
```

## Project Structure

```
ire-project/
├── baseline.py              # Collect interactions from simulator
├── indexing.py              # Index articles to Elasticsearch
├── scripts/
│   ├── extract_features.py  # Feature engineering
│   ├── train.py             # Train all personalization models
│   └── evaluate.py          # A/B testing against simulator
├── src/
│   ├── features/            # Feature extractors
│   │   ├── user_features.py
│   │   ├── article_features.py
│   │   └── graph_builder.py
│   ├── logging/             # Log reader & metrics
│   │   ├── interaction_logger.py
│   │   └── metrics_calculator.py
│   └── models/              # Personalization models
│       ├── gnn.py           # Graph Neural Network
│       ├── ltr.py           # Learning to Rank
│       └── reranker.py      # Hybrid reranker
└── data/
    ├── raw/                 # interaction_logs.jsonl
    └── processed/           # Features & trained models
```

## Detailed Steps

### 1. Setup
```bash
docker compose up -d
pip install -r requirements.txt
python indexing.py  # Only needed once
```

### 2. Data Collection
```bash
python baseline.py  # Runs 100 iterations, logs to interaction_logs.jsonl
```

### 3. Feature Engineering
```bash
python scripts/extract_features.py
```
Outputs:
- `data/processed/user_features.json` - User topic preferences, CTR patterns
- `data/processed/article_features.json` - Article content features
- `data/processed/interaction_graph.pkl` - User-Article-Topic graph for GNN
- `data/processed/baseline_metrics.json` - Baseline CTR/NDCG scores

### 4. Training
```bash
python scripts/train.py
```
Trains:
- GNN model on interaction graph
- LTR model on user-article feature pairs

### 5. Evaluation
```bash
python scripts/evaluate.py
```
Runs personalized ranking against simulator and reports improvement over baseline.

## Reset Environment
```bash
docker compose down -v
docker compose up -d
python indexing.py
```
