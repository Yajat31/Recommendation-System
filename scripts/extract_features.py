"""
Feature Engineering Script
Processes interaction logs from baseline.py and extracts features for personalization
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import logging
import pickle
from pathlib import Path
from collections import defaultdict

from src.logging.interaction_logger import InteractionLogReader
from src.logging.metrics_calculator import MetricsCalculator
from src.features.user_features import UserFeatureExtractor
from src.features.article_features import ArticleFeatureExtractor
from src.features.graph_builder import UserArticleGraphBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FeatureEngineer:
    """Process interaction logs and extract features"""
    
    def __init__(self, log_path: str = "interaction_logs.jsonl", articles_path: str = "artifacts/articles.jsonl"):
        self.log_reader = InteractionLogReader(log_path)
        self.user_extractor = UserFeatureExtractor()
        self.article_extractor = ArticleFeatureExtractor(articles_path=articles_path)
        self.graph_builder = UserArticleGraphBuilder()
        
    def load_interactions(self):
        """Load interaction logs"""
        logging.info("Loading interaction logs...")
        interactions = self.log_reader.load_interactions()
        logging.info(f"Loaded {len(interactions)} interactions")
        return interactions
    
    def extract_user_features(self, interactions):
        """Extract features for all users"""
        logging.info("Extracting user features...")
        
        user_interactions = defaultdict(list)
        for interaction in interactions:
            user_interactions[interaction['user_id']].append(interaction)
        
        logging.info(f"Found {len(user_interactions)} unique users")
        
        user_features = {}
        for user_id, user_ints in user_interactions.items():
            user_features[user_id] = self.user_extractor.extract_features(user_id, user_ints)
        
        output_path = Path("data/processed/user_features.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(user_features, f, indent=2)
        
        logging.info(f"User features saved to {output_path}")
        return user_features
    
    def extract_article_features(self, interactions):
        """Extract features for all articles in interactions"""
        logging.info("Extracting article features...")
        
        article_ids = set()
        for interaction in interactions:
            for action in interaction['actions']:
                article_ids.add(action['article_id'])
        
        logging.info(f"Found {len(article_ids)} unique articles")
        
        article_features = {}
        for article_id in article_ids:
            article_features[article_id] = self.article_extractor.extract_features(article_id, interactions)
        
        output_path = Path("data/processed/article_features.json")
        with open(output_path, 'w') as f:
            json.dump(article_features, f, indent=2)
        
        logging.info(f"Article features saved to {output_path}")
        return article_features
    
    def build_interaction_graph(self, interactions):
        """Build user-article-topic graph for GNN"""
        logging.info("Building interaction graph...")
        
        article_metadata = {}
        for interaction in interactions:
            for action in interaction['actions']:
                article_id = action['article_id']
                if article_id not in article_metadata:
                    article_metadata[article_id] = self.article_extractor.get_article_metadata(article_id)
        
        graph = self.graph_builder.build_graph(interactions, article_metadata)
        
        # Save graph
        output_path = Path("data/processed/interaction_graph.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph_builder, f)
        logging.info(f"Graph saved to {output_path}")
        
        # Export GNN format
        gnn_data = self.graph_builder.export_for_gnn()
        gnn_path = Path("data/processed/graph_gnn_format.pkl")
        with open(gnn_path, 'wb') as f:
            pickle.dump(gnn_data, f)
        logging.info(f"GNN-format graph saved to {gnn_path}")
        
        return graph
    
    def compute_baseline_metrics(self, interactions):
        """Compute baseline metrics"""
        logging.info("Computing baseline metrics...")
        
        metrics = MetricsCalculator.get_all_metrics(interactions)
        
        logging.info("\n=== Baseline Metrics ===")
        logging.info(f"CTR: {metrics['ctr']:.4f}")
        logging.info(f"MRR: {metrics['mrr']:.4f}")
        logging.info(f"NDCG@5: {metrics['ndcg@5']:.4f}")
        logging.info(f"NDCG@10: {metrics['ndcg@10']:.4f}")
        logging.info(f"Engagement Rate: {metrics['engagement_rate']:.4f}")
        logging.info(f"Avg Dwell Time: {metrics['avg_dwell_time']:.2f}s")
        logging.info(f"Skip Rate: {metrics['skip_rate']:.4f}")
        logging.info("========================")
        
        output_path = Path("data/processed/baseline_metrics.json")
        metrics_json = {k: (dict(v) if isinstance(v, dict) else v) for k, v in metrics.items()}
        with open(output_path, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        logging.info(f"Metrics saved to {output_path}")
        return metrics


def main():
    logging.info("=" * 50)
    logging.info("Feature Engineering")
    logging.info("=" * 50)
    
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    
    engineer = FeatureEngineer()
    
    interactions = engineer.load_interactions()
    if not interactions:
        logging.error("No interactions found! Run baseline.py first.")
        return
    
    engineer.extract_user_features(interactions)
    engineer.extract_article_features(interactions)
    engineer.build_interaction_graph(interactions)
    engineer.compute_baseline_metrics(interactions)
    
    logging.info("\n" + "=" * 50)
    logging.info("Feature engineering complete!")
    logging.info("=" * 50)
    logging.info("\nGenerated files:")
    logging.info("  - data/processed/user_features.json")
    logging.info("  - data/processed/article_features.json")
    logging.info("  - data/processed/interaction_graph.pkl")
    logging.info("  - data/processed/graph_gnn_format.pkl")
    logging.info("  - data/processed/baseline_metrics.json")


if __name__ == "__main__":
    main()
