"""
Train personalization models (GNN + LTR + CF + Content + Ensemble + PositionAware)
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
from pathlib import Path

from src.logging.interaction_logger import InteractionLogReader
from src.models.gnn import GNNRanker
from src.models.ltr import LTRRanker
from src.models.collaborative import CollaborativeFilteringEnsemble
from src.models.content_based import ContentBasedFilter, PopularityRanker
from src.models.reranker import EnsembleReranker, PositionAwareReranker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    logging.info("=" * 60)
    logging.info("Training Personalization Models (Full Pipeline)")
    logging.info("=" * 60)
    
    # Load interactions
    log_reader = InteractionLogReader("interaction_logs.jsonl")
    interactions = log_reader.load_interactions()
    
    if not interactions:
        logging.error("No interactions found! Run baseline.py first.")
        return
    
    logging.info(f"Loaded {len(interactions)} interactions")
    
    # Count positive interactions
    positive = sum(1 for i in interactions 
                   for a in i.get('actions', []) 
                   if a.get('clicked', False))
    logging.info(f"Positive interactions (clicks): {positive}")
    
    # Train individual models
    logging.info("\n" + "=" * 60)
    logging.info("PHASE 1: Training Individual Models")
    logging.info("=" * 60)
    
    # 1. GNN
    logging.info("\n--- [1/5] GNN Model ---")
    gnn = GNNRanker(embedding_dim=64, num_iterations=4, use_attention=True, alpha=0.6)
    gnn.train("data/processed/interaction_graph.pkl")
    gnn.save("data/processed/gnn_model.pkl")
    logging.info(f"GNN: {len(gnn.user_embeddings)} users, {len(gnn.article_embeddings)} articles")
    
    # 2. LTR
    logging.info("\n--- [2/5] LTR Model ---")
    ltr = LTRRanker()
    ltr.train(interactions)
    ltr.save("data/processed/ltr_model.pkl")
    logging.info(f"LTR: trained with {len(interactions)} samples")
    
    # 3. Collaborative Filtering
    logging.info("\n--- [3/5] Collaborative Filtering ---")
    cf = CollaborativeFilteringEnsemble(user_cf_weight=0.3, item_cf_weight=0.3, mf_weight=0.4)
    cf.train(interactions)
    cf.save("data/processed/cf_model.pkl")
    
    # 4. Content-Based
    logging.info("\n--- [4/5] Content-Based Filter ---")
    content = ContentBasedFilter(top_k_terms=100)
    content.train(interactions, "data/processed/article_features.json")
    content.save("data/processed/content_model.pkl")
    
    # 5. Popularity
    logging.info("\n--- [5/5] Popularity Ranker ---")
    popularity = PopularityRanker()
    popularity.train(interactions)
    popularity.save("data/processed/popularity_model.pkl")
    
    # Train ensemble
    logging.info("\n" + "=" * 60)
    logging.info("PHASE 2: Training Ensemble Reranker")
    logging.info("=" * 60)

    # Use updated weights that favor best performers (CF, Content)
    ensemble = EnsembleReranker(weights={
        'cf': 0.35,       # Best performer (+22.2% CTR)
        'content': 0.25,  # Second best (+11.1% CTR)
        'popularity': 0.15,
        'gnn': 0.15,
        'ltr': 0.10
    })
    ensemble.train(
        interactions,
        "data/processed/interaction_graph.pkl",
        "data/processed/user_features.json",
        "data/processed/article_features.json"
    )
    ensemble.save("data/processed/ensemble_model.pkl")
    
    # Train position-aware reranker
    logging.info("\n" + "=" * 60)
    logging.info("PHASE 3: Training Position-Aware Reranker")
    logging.info("=" * 60)
    
    pos_aware = PositionAwareReranker(personalization_strength=0.4)
    pos_aware.train(interactions)
    pos_aware.save("data/processed/position_aware_model.pkl")

    logging.info("\n" + "=" * 60)
    logging.info("Training complete!")
    logging.info("=" * 60)
    logging.info("\nSaved models:")
    logging.info("  - data/processed/gnn_model.pkl")
    logging.info("  - data/processed/ltr_model.pkl")
    logging.info("  - data/processed/cf_model.pkl")
    logging.info("  - data/processed/content_model.pkl")
    logging.info("  - data/processed/popularity_model.pkl")
    logging.info("  - data/processed/ensemble_model.pkl")
    logging.info("  - data/processed/position_aware_model.pkl")
    logging.info("\nNext: Run evaluate.py to test against simulator")
if __name__ == "__main__":
    main()
