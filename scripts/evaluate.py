"""
Evaluate personalization models against the simulator
Comprehensive A/B testing of all ranking strategies
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import time
import logging
import requests
from datetime import datetime
from pathlib import Path
from elasticsearch import Elasticsearch

from src.logging.metrics_calculator import MetricsCalculator
from src.models.gnn import GNNRanker
from src.models.ltr import LTRRanker
from src.models.collaborative import CollaborativeFilteringEnsemble
from src.models.content_based import ContentBasedFilter, PopularityRanker
from src.models.reranker import EnsembleReranker, HybridReranker, PositionAwareReranker
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SIMULATOR_URL = "http://localhost:3000"
ES_HOST = "http://localhost:9200"
INDEX_NAME = "articles"
NUM_EVAL_ITERATIONS = 200  # More iterations for stable results

# Candidate pool settings
CANDIDATE_POOL_SIZE = 50  # Fetch more candidates for personalization
FINAL_RESULT_SIZE = 10    # Return this many results


class OnlineAdaptiveReranker:
    """
    Adaptive reranker that learns user preferences during evaluation.
    Starts with baseline or popularity for cold-start users, then
    adapts as we see more interactions from the same user.
    """
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
        # Track user sessions during evaluation
        self.session_users = {}  # user_id -> list of (article_id, clicked)
        self.session_clicks = {}  # article_id -> click count
        self.session_impressions = {}  # article_id -> impression count
        
    def get_user_familiarity(self, user_id: str) -> float:
        """
        Calculate how familiar we are with a user.
        Returns value between 0 (cold-start) and 1 (well-known)
        """
        # Check if user is in training data
        in_training = False
        if self.evaluator.gnn and hasattr(self.evaluator.gnn, 'user_embeddings'):
            in_training = user_id in self.evaluator.gnn.user_embeddings
        
        # Check session history
        session_history = len(self.session_users.get(user_id, []))
        
        if in_training:
            return min(1.0, 0.7 + session_history * 0.1)  # Known user
        else:
            return min(0.6, session_history * 0.15)  # New user, learn from session
    
    def update_session(self, user_id: str, article_ids: list, actions: list):
        """Update session data after receiving feedback"""
        if user_id not in self.session_users:
            self.session_users[user_id] = []
        
        for i, article_id in enumerate(article_ids):
            # Track impressions
            self.session_impressions[article_id] = self.session_impressions.get(article_id, 0) + 1
            
            # Track clicks
            if i < len(actions):
                action = actions[i]
                clicked = action.get('clicked', False) if isinstance(action, dict) else False
                self.session_users[user_id].append((article_id, clicked))
                if clicked:
                    self.session_clicks[article_id] = self.session_clicks.get(article_id, 0) + 1
    
    def get_session_scores(self, article_ids: list) -> np.ndarray:
        """Score articles based on session popularity"""
        scores = np.zeros(len(article_ids))
        for i, aid in enumerate(article_ids):
            impressions = self.session_impressions.get(aid, 0)
            clicks = self.session_clicks.get(aid, 0)
            if impressions > 0:
                scores[i] = clicks / impressions  # Session CTR
            else:
                # Use training popularity if available
                if self.evaluator.popularity:
                    scores[i] = self.evaluator.popularity.predict(aid)
        return scores
    
    def rerank(self, user_id: str, article_ids: list) -> list:
        """
        Adaptive reranking that blends personalization with baseline.
        For cold-start: use baseline position + popularity boost
        For known users: use full personalization
        """
        familiarity = self.get_user_familiarity(user_id)
        n = len(article_ids)
        
        # Baseline scores (position-based - higher position = higher score)
        baseline_scores = np.array([(n - i) / n for i in range(n)])
        
        # Get personalized scores if available
        personalized_scores = np.zeros(n)
        if familiarity > 0.3:
            # Use ensemble scoring
            try:
                if self.evaluator.ensemble:
                    raw_scores = self.evaluator.ensemble.score(user_id, article_ids)
                    personalized_scores = (np.array(raw_scores) - np.min(raw_scores))
                    if np.max(personalized_scores) > 0:
                        personalized_scores /= np.max(personalized_scores)
            except:
                pass
        
        # Session-based popularity boost
        session_scores = self.get_session_scores(article_ids)
        
        # Blend based on familiarity
        # Cold-start: baseline + session_popularity
        # Known: personalization + small baseline anchor
        final_scores = (
            (1 - familiarity) * 0.6 * baseline_scores +      # Anchor to baseline
            (1 - familiarity) * 0.4 * session_scores +       # Session popularity
            familiarity * 0.8 * personalized_scores +        # Personalization
            familiarity * 0.2 * baseline_scores              # Keep some relevance
        )
        
        # Sort by final scores
        ranked_indices = np.argsort(final_scores)[::-1]
        return [article_ids[i] for i in ranked_indices]


class Evaluator:
    """Evaluate different ranking strategies for A/B testing"""
    
    def __init__(self):
        self.es = Elasticsearch(hosts=[ES_HOST])
        
        # Check ES connection
        if not self.es.ping():
            raise ConnectionError("Elasticsearch is not reachable!")
        logging.info("Elasticsearch connected")
        
        # Check simulator
        try:
            resp = requests.get(f"{SIMULATOR_URL}/query", timeout=5)
            if resp.status_code != 200:
                raise ConnectionError(f"Simulator returned {resp.status_code}")
            logging.info("Simulator connected")
        except Exception as e:
            raise ConnectionError(f"Simulator not reachable: {e}")
        
        # Load all models
        self._load_models()
    
    def _load_models(self):
        """Load all available models"""
        logging.info("\nLoading models...")
        
        # GNN
        self.gnn = GNNRanker()
        try:
            self.gnn.load("data/processed/gnn_model.pkl")
            logging.info("  ✓ GNN loaded")
        except Exception as e:
            logging.warning(f"  ✗ GNN not loaded: {e}")
            self.gnn = None
        
        # LTR
        self.ltr = LTRRanker()
        try:
            self.ltr.load("data/processed/ltr_model.pkl")
            self.ltr.load_features()
            logging.info("  ✓ LTR loaded")
        except Exception as e:
            logging.warning(f"  ✗ LTR not loaded: {e}")
            self.ltr = None
        
        # Collaborative Filtering
        try:
            self.cf = CollaborativeFilteringEnsemble.load("data/processed/cf_model.pkl")
            logging.info("  ✓ Collaborative Filtering loaded")
        except Exception as e:
            logging.warning(f"  ✗ CF not loaded: {e}")
            self.cf = None
        
        # Content-Based
        try:
            self.content = ContentBasedFilter.load("data/processed/content_model.pkl")
            logging.info("  ✓ Content-Based loaded")
        except Exception as e:
            logging.warning(f"  ✗ Content-Based not loaded: {e}")
            self.content = None
        
        # Popularity
        try:
            self.popularity = PopularityRanker.load("data/processed/popularity_model.pkl")
            logging.info("  ✓ Popularity loaded")
        except Exception as e:
            logging.warning(f"  ✗ Popularity not loaded: {e}")
            self.popularity = None
        
        # Ensemble
        try:
            self.ensemble = EnsembleReranker.load("data/processed/ensemble_model.pkl")
            logging.info("  ✓ Ensemble loaded")
        except Exception as e:
            logging.warning(f"  ✗ Ensemble not loaded: {e}")
            # Fallback: create ensemble from individual models
            self.ensemble = None
            if self.gnn and self.ltr:
                self.hybrid = HybridReranker()
                self.hybrid.gnn = self.gnn
                self.hybrid.ltr = self.ltr
                self.hybrid.trained = True
                logging.info("  ✓ Hybrid (fallback) created")
            else:
                self.hybrid = None
        
        # Position-Aware Reranker
        try:
            self.pos_aware = PositionAwareReranker.load("data/processed/position_aware_model.pkl")
            logging.info("  ✓ Position-Aware loaded")
        except Exception as e:
            logging.warning(f"  ✗ Position-Aware not loaded: {e}")
            self.pos_aware = None
    
    def get_query(self):
        """Get query from simulator"""
        try:
            response = requests.get(f"{SIMULATOR_URL}/query", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logging.error(f"Query error: {e}")
        return None
    
    def search_es(self, query_text: str, size: int = CANDIDATE_POOL_SIZE):
        """Get baseline ES results with expanded candidate pool"""
        body = {
            "query": {
                "multi_match": {
                    "query": query_text,
                    "fields": ["text^2", "topics"],
                    "type": "best_fields"
                }
            },
            "size": size
        }
        try:
            response = self.es.search(index=INDEX_NAME, body=body)
            return [hit['_id'] for hit in response['hits']['hits']]
        except Exception as e:
            logging.error(f"ES search error: {e}")
            return []
    
    def get_popular_articles(self, n: int = 20) -> list:
        """Get top popular articles to inject diversity"""
        if self.popularity and hasattr(self.popularity, 'article_scores'):
            # Sort by popularity score
            sorted_articles = sorted(
                self.popularity.article_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return [aid for aid, _ in sorted_articles[:n]]
        return []
    
    def expand_candidates_with_diversity(self, es_results: list, user_id: str) -> list:
        """
        Expand candidate pool with diverse articles.
        Key insight: Sparse data means we need to expose users to more variety.
        """
        candidates = list(es_results)  # Start with ES results
        seen = set(candidates)
        
        # 1. Add popular articles (helps cold-start)
        popular = self.get_popular_articles(20)
        for aid in popular:
            if aid not in seen:
                candidates.append(aid)
                seen.add(aid)
                if len(candidates) >= CANDIDATE_POOL_SIZE + 10:
                    break
        
        # 2. If user has history in CF, add similar items
        if self.cf and hasattr(self.cf, 'item_cf') and self.cf.item_cf:
            try:
                # Get user's previously clicked items from user_cf
                user_articles = self.cf.user_cf.user_article_matrix.get(user_id, {})
                clicked_items = list(user_articles.keys())[:5]  # Top 5 clicked
                for clicked_item in clicked_items:
                    # Find similar items
                    similar = self.cf.item_cf.get_similar_items(clicked_item, n=5)
                    for similar_id, _ in similar:
                        if similar_id not in seen:
                            candidates.append(similar_id)
                            seen.add(similar_id)
            except Exception as e:
                pass
        
        return candidates
    
    def submit_ranklist(self, query_id: str, user_id: str, ranked_ids: list):
        """Submit ranking and get feedback"""
        payload = {
            "query_id": query_id,
            "user_id": user_id,
            "ranked_article_ids": ranked_ids
        }
        try:
            response = requests.post(f"{SIMULATOR_URL}/ranklist", json=payload, timeout=10)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logging.error(f"Submit error: {e}")
        return None
    
    def parse_actions(self, ranked_ids: list, raw_actions: list):
        """Parse raw actions into structured format"""
        parsed_actions = []
        for idx, article_id in enumerate(ranked_ids):
            article_actions = raw_actions[idx] if idx < len(raw_actions) else []
            
            action_dict = {
                'article_id': article_id,
                'position': idx,
                'clicked': False,
                'dwell_time_secs': 0.0,
                'liked': False,
                'shared': False,
                'bookmarked': False,
                'skipped': len(article_actions) == 0
            }
            
            for action in article_actions:
                if action == "Click":
                    action_dict['clicked'] = True
                elif isinstance(action, dict) and "Dwell" in action:
                    dwell = action["Dwell"]
                    action_dict['dwell_time_secs'] = dwell['secs'] + dwell['nanos'] / 1e9
                elif action == "Like":
                    action_dict['liked'] = True
                elif action == "Share":
                    action_dict['shared'] = True
                elif action == "Bookmark":
                    action_dict['bookmarked'] = True
            
            parsed_actions.append(action_dict)
        
        return parsed_actions
    
    def rerank(self, strategy: str, user_id: str, article_ids: list, 
               adaptive_reranker: 'OnlineAdaptiveReranker' = None,
               return_size: int = FINAL_RESULT_SIZE) -> list:
        """Apply reranking strategy and return top results"""
        try:
            if strategy == "baseline":
                # Baseline just uses ES order, truncated
                return article_ids[:return_size]
            elif strategy == "adaptive" and adaptive_reranker:
                reranked = adaptive_reranker.rerank(user_id, article_ids)
            elif strategy == "gnn" and self.gnn:
                reranked = self.gnn.rerank(user_id, article_ids)
            elif strategy == "ltr" and self.ltr:
                reranked = self.ltr.rerank(user_id, article_ids)
            elif strategy == "cf" and self.cf:
                reranked = self.cf.rerank(user_id, article_ids)
            elif strategy == "content" and self.content:
                reranked = self.content.rerank(user_id, article_ids)
            elif strategy == "popularity" and self.popularity:
                reranked = self.popularity.rerank(article_ids)
            elif strategy == "ensemble" and self.ensemble:
                reranked = self.ensemble.rerank(user_id, article_ids)
            elif strategy == "hybrid" and self.hybrid:
                reranked = self.hybrid.rerank(user_id, article_ids)
            elif strategy == "pos_aware" and self.pos_aware:
                reranked = self.pos_aware.rerank(user_id, article_ids)
            elif strategy == "expanded":
                # New strategy: expanded pool + ensemble reranking
                reranked = self._rerank_expanded(user_id, article_ids)
            elif strategy == "smart":
                # Smart strategy: blend position + content + popularity
                reranked = self._rerank_smart(user_id, article_ids)
            else:
                return article_ids[:return_size]
            
            # Always return only top N results
            return reranked[:return_size]
        except Exception as e:
            logging.warning(f"Rerank failed for {strategy}: {e}")
            return article_ids[:return_size]
    
    def _rerank_expanded(self, user_id: str, article_ids: list) -> list:
        """
        Smart reranking for expanded candidate pool.
        Combines multiple signals with position-aware scoring.
        """
        n = len(article_ids)
        scores = []
        
        for i, article_id in enumerate(article_ids):
            # 1. Position score from ES (relevance signal)
            # Exponential decay: position 1 = 1.0, position 50 = ~0.007
            es_position_score = np.exp(-i * 0.1)
            
            # 2. Popularity score
            pop_score = 0.0
            if self.popularity:
                pop_score = self.popularity.predict(article_id)
            
            # 3. CF personalization score
            cf_score = 0.0
            if self.cf:
                cf_score = self.cf.predict(user_id, article_id)
            
            # 4. Content similarity score  
            content_score = 0.0
            if self.content:
                content_score = self.content.predict(user_id, article_id)
            
            # Combine with adaptive weighting
            # Key: ES relevance is important, personalization boosts it
            final_score = (
                0.35 * es_position_score +   # ES relevance anchor
                0.25 * pop_score +            # Popularity (proven to work)
                0.25 * cf_score +             # Collaborative filtering
                0.15 * content_score          # Content similarity
            )
            
            scores.append((article_id, final_score))
        
        # Sort by combined score
        scores.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scores]
    
    def _rerank_smart(self, user_id: str, article_ids: list) -> list:
        """
        Smart reranking based on what works best:
        - Content-Based (topic matching) is very effective
        - Position from ES provides relevance anchor
        - Popularity helps with cold-start
        """
        n = len(article_ids)
        scores = []
        
        for i, article_id in enumerate(article_ids):
            # 1. ES position score (relevance)
            # Gentler decay to not lose good candidates
            es_score = 1.0 / (1 + i * 0.1)  # Position 1=1.0, 10=0.5, 50=0.17
            
            # 2. Content score (best performer!)
            content_score = 0.0
            if self.content:
                content_score = self.content.predict(user_id, article_id)
            
            # 3. Popularity as tiebreaker
            pop_score = 0.0
            if self.popularity:
                pop_score = self.popularity.predict(article_id)
            
            # Weight content heavily since it performs best
            final_score = (
                0.30 * es_score +        # ES relevance
                0.50 * content_score +   # Content matching (key!)
                0.20 * pop_score         # Popularity
            )
            
            scores.append((article_id, final_score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scores]
    
    def run_evaluation(self, strategy: str, num_iterations: int = NUM_EVAL_ITERATIONS,
                       adaptive_reranker: 'OnlineAdaptiveReranker' = None):
        """Run evaluation with specified strategy"""
        logging.info(f"\n  Evaluating: {strategy.upper()}")
        
        interactions = []
        successful = 0
        
        for i in range(num_iterations):
            query_data = self.get_query()
            if not query_data:
                continue
            
            # Get ES results - fetch CANDIDATE_POOL_SIZE for all strategies (FAIR COMPARISON)
            baseline_ids = self.search_es(query_data['query_text'])
            if not baseline_ids:
                continue
            
            # FIXED: All strategies now get the same candidate pool for fair comparison
            # Expand candidate pool with diversity for ALL strategies including baseline
            if strategy == "baseline":
                # Baseline: use ES results directly without reranking, but same pool size
                candidates = baseline_ids[:CANDIDATE_POOL_SIZE]
            else:
                # Personalized: expand with diversity before reranking
                candidates = self.expand_candidates_with_diversity(baseline_ids, query_data['user_id'])
            
            ranked_ids = self.rerank(strategy, query_data['user_id'], candidates, 
                                     adaptive_reranker=adaptive_reranker)
            
            result = self.submit_ranklist(query_data['query_id'], query_data['user_id'], ranked_ids)
            if not result:
                continue
            
            raw_actions = result.get('actions', [])
            
            # Validate response
            if len(raw_actions) != len(ranked_ids):
                logging.warning(f"Action count mismatch: sent {len(ranked_ids)} articles, got {len(raw_actions)} actions")
            
            parsed_actions = self.parse_actions(ranked_ids, raw_actions)
            
            # Update adaptive reranker with feedback (online learning)
            if adaptive_reranker and strategy == "adaptive":
                adaptive_reranker.update_session(query_data['user_id'], ranked_ids, parsed_actions)
            
            interactions.append({
                'query_id': query_data['query_id'],
                'user_id': query_data['user_id'],
                'query_text': query_data['query_text'],
                'strategy': strategy,
                'actions': parsed_actions
            })
            
            successful += 1
            time.sleep(0.05)
        
        # Calculate metrics
        if interactions:
            metrics = MetricsCalculator.get_all_metrics(interactions)
            metrics['iterations'] = successful
        else:
            metrics = {
                'ctr': 0.0, 'mrr': 0.0, 'ndcg@5': 0.0, 'ndcg@10': 0.0,
                'engagement_rate': 0.0, 'avg_dwell_time': 0.0, 'skip_rate': 0.0,
                'position_ctr': {}, 'iterations': 0
            }
        
        return metrics, interactions


def main():
    logging.info("=" * 70)
    logging.info("A/B Testing: Personalization Model Evaluation")
    logging.info("=" * 70)
    
    try:
        evaluator = Evaluator()
    except Exception as e:
        logging.error(f"Failed to initialize: {e}")
        return
    
    # Create adaptive reranker for online learning
    adaptive_reranker = OnlineAdaptiveReranker(evaluator)
    
    # Define strategies to test (including GNN for graph-based personalization)
    strategies = ["baseline", "gnn", "smart", "content", "pos_aware", "ensemble", "popularity"]
    
    # Add other strategies for comparison
    if evaluator.cf:
        strategies.append("cf")
    
    strategies.append("adaptive")  # Online learning
    
    results = {}
    all_interactions = {}
    
    logging.info(f"\nTesting {len(strategies)} strategies with {NUM_EVAL_ITERATIONS} iterations each...")
    
    for strategy in strategies:
        # Pass adaptive_reranker for adaptive strategy
        metrics, interactions = evaluator.run_evaluation(
            strategy, 
            num_iterations=NUM_EVAL_ITERATIONS,
            adaptive_reranker=adaptive_reranker if strategy == "adaptive" else None
        )
        results[strategy] = metrics
        all_interactions[strategy] = interactions
        
        ctr = metrics.get('ctr', 0)
        mrr = metrics.get('mrr', 0)
        logging.info(f"    → CTR: {ctr:.4f}, MRR: {mrr:.4f}")
    
    # Print comparison table
    logging.info("\n" + "=" * 90)
    logging.info("RESULTS COMPARISON (A/B Test)")
    logging.info("=" * 90)
    
    header = f"{'Metric':<18}"
    for s in strategies:
        header += f" {s.upper():>10}"
    logging.info(header)
    logging.info("-" * 90)
    
    for metric in ['ctr', 'mrr', 'ndcg@5', 'ndcg@10', 'engagement_rate', 'avg_dwell_time']:
        row = f"{metric:<18}"
        for strategy in strategies:
            val = results[strategy].get(metric, 0)
            row += f" {val:>10.4f}"
        logging.info(row)
    
    logging.info("-" * 90)
    
    # Calculate improvements vs baseline
    baseline_ctr = results['baseline']['ctr']
    baseline_mrr = results['baseline']['mrr']
    
    logging.info("\nImprovement vs Baseline:")
    logging.info("-" * 50)
    
    for strategy in strategies:
        if strategy == "baseline":
            continue
        
        ctr_imp = ((results[strategy]['ctr'] - baseline_ctr) / baseline_ctr * 100) if baseline_ctr > 0 else 0
        mrr_imp = ((results[strategy]['mrr'] - baseline_mrr) / baseline_mrr * 100) if baseline_mrr > 0 else 0
        
        logging.info(f"  {strategy.upper():12} → CTR: {ctr_imp:+6.1f}%  |  MRR: {mrr_imp:+6.1f}%")
    
    # Find best model
    best_strategy = max(strategies, key=lambda s: results[s]['ctr'])
    best_ctr = results[best_strategy]['ctr']
    best_improvement = ((best_ctr - baseline_ctr) / baseline_ctr * 100) if baseline_ctr > 0 else 0
    
    logging.info("\n" + "=" * 90)
    logging.info(f"🏆 BEST MODEL: {best_strategy.upper()} (CTR: {best_ctr:.4f}, +{best_improvement:.1f}% vs baseline)")
    logging.info("=" * 90)
    
    # Save comprehensive results
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_iterations': NUM_EVAL_ITERATIONS,
            'strategies': strategies
        },
        'metrics': {},
        'improvements': {}
    }
    
    for strategy in strategies:
        output['metrics'][strategy] = {
            k: (v if not isinstance(v, dict) else {str(kk): vv for kk, vv in v.items()})
            for k, v in results[strategy].items()
        }
        
        if strategy != "baseline":
            output['improvements'][strategy] = {
                'ctr': ((results[strategy]['ctr'] - baseline_ctr) / baseline_ctr * 100) if baseline_ctr > 0 else 0,
                'mrr': ((results[strategy]['mrr'] - baseline_mrr) / baseline_mrr * 100) if baseline_mrr > 0 else 0
            }
    
    output['best_model'] = {
        'strategy': best_strategy,
        'ctr': best_ctr,
        'improvement': best_improvement
    }
    
    output_path = Path("data/processed/ab_test_results.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    logging.info(f"\nResults saved to {output_path}")
    
    # Also save to artifacts for reporting
    artifacts_path = Path("artifacts/evaluation")
    artifacts_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_file = artifacts_path / f"ab_test_{timestamp}.json"
    with open(artifact_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    logging.info(f"Artifact saved to {artifact_file}")


if __name__ == "__main__":
    main()
