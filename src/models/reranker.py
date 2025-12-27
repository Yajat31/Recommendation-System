"""
Advanced reranker combining multiple personalization models
Supports GNN, LTR, Collaborative Filtering, and Content-Based filtering
"""
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional
from .gnn import GNNRanker
from .ltr import LTRRanker
from .collaborative import CollaborativeFilteringEnsemble
from .content_based import ContentBasedFilter, PopularityRanker

logging.basicConfig(level=logging.INFO)


class EnsembleReranker:
    """
    Advanced ensemble combining all personalization models
    Supports dynamic weight tuning and A/B testing
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize with optional custom weights
        
        Args:
            weights: Dict mapping model names to weights
                     Keys: 'gnn', 'ltr', 'cf', 'content', 'popularity'
        """
        self.gnn = GNNRanker(embedding_dim=64, num_iterations=4, use_attention=True)
        self.ltr = LTRRanker()
        self.cf = CollaborativeFilteringEnsemble()
        self.content = ContentBasedFilter()
        self.popularity = PopularityRanker()
        
        # Updated weights based on A/B test results - favor CF and Content
        self.weights = weights or {
            'cf': 0.40,          # Best performer - increased from 0.25
            'content': 0.25,     # Second best - increased from 0.15
            'popularity': 0.20,  # Good for cold-start - increased from 0.10
            'ltr': 0.10,         # Reduced from 0.15
            'gnn': 0.05          # Reduced from 0.35 (underperforming)
        }
        
        self.model_scores = {}  # Track individual model contributions
        self.trained = False
        
    def train(self, interactions: List[Dict],
              graph_path: str = "data/processed/interaction_graph.pkl",
              user_features_path: str = "data/processed/user_features.json",
              article_features_path: str = "data/processed/article_features.json"):
        """Train all models"""
        logging.info("=" * 60)
        logging.info("Training Ensemble Reranker (5 models)")
        logging.info("=" * 60)
        
        # Train GNN
        logging.info("\n[1/5] Training GNN model...")
        try:
            self.gnn.train(graph_path)
        except Exception as e:
            logging.warning(f"GNN training failed: {e}")
        
        # Train LTR
        logging.info("\n[2/5] Training LTR model...")
        try:
            self.ltr.load_features(user_features_path, article_features_path)
            self.ltr.train(interactions, user_features_path, article_features_path)
        except Exception as e:
            logging.warning(f"LTR training failed: {e}")
        
        # Train CF
        logging.info("\n[3/5] Training Collaborative Filtering...")
        try:
            self.cf.train(interactions)
        except Exception as e:
            logging.warning(f"CF training failed: {e}")
        
        # Train Content-Based
        logging.info("\n[4/5] Training Content-Based Filter...")
        try:
            self.content.train(interactions, article_features_path)
        except Exception as e:
            logging.warning(f"Content training failed: {e}")
        
        # Train Popularity
        logging.info("\n[5/5] Training Popularity Ranker...")
        try:
            self.popularity.train(interactions)
        except Exception as e:
            logging.warning(f"Popularity training failed: {e}")
        
        self.trained = True
        logging.info("\n" + "=" * 60)
        logging.info("Ensemble training complete!")
        logging.info(f"Weights: {self.weights}")
        logging.info("=" * 60)
        return self
    
    def _normalize_scores(self, scores: List[float]) -> np.ndarray:
        """Normalize scores to [0, 1] range"""
        scores = np.array(scores)
        if len(scores) == 0:
            return scores
        min_s, max_s = scores.min(), scores.max()
        if max_s != min_s:
            return (scores - min_s) / (max_s - min_s)
        return np.ones_like(scores) * 0.5
    
    def score(self, user_id: str, article_ids: List[str]) -> List[float]:
        """
        Combine all model scores with learned weights
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        n = len(article_ids)
        final_scores = np.zeros(n)
        self.model_scores = {k: np.zeros(n) for k in self.weights.keys()}
        
        # Track which models contributed
        active_models = []
        
        # GNN scores
        if self.gnn.trained and self.weights.get('gnn', 0) > 0:
            try:
                gnn_raw = self.gnn.score(user_id, article_ids)
                self.model_scores['gnn'] = self._normalize_scores(gnn_raw)
                final_scores += self.weights['gnn'] * self.model_scores['gnn']
                active_models.append('gnn')
            except Exception as e:
                logging.warning(f"GNN scoring failed: {e}")
        
        # LTR scores
        if self.ltr.trained and self.weights.get('ltr', 0) > 0:
            try:
                ltr_raw = self.ltr.score(user_id, article_ids)
                self.model_scores['ltr'] = self._normalize_scores(ltr_raw)
                final_scores += self.weights['ltr'] * self.model_scores['ltr']
                active_models.append('ltr')
            except Exception as e:
                logging.warning(f"LTR scoring failed: {e}")
        
        # CF scores
        if self.cf.trained and self.weights.get('cf', 0) > 0:
            try:
                cf_raw = [self.cf.predict(user_id, aid) for aid in article_ids]
                self.model_scores['cf'] = self._normalize_scores(cf_raw)
                final_scores += self.weights['cf'] * self.model_scores['cf']
                active_models.append('cf')
            except Exception as e:
                logging.warning(f"CF scoring failed: {e}")
        
        # Content scores
        if self.content.trained and self.weights.get('content', 0) > 0:
            try:
                content_raw = [self.content.predict(user_id, aid) for aid in article_ids]
                self.model_scores['content'] = self._normalize_scores(content_raw)
                final_scores += self.weights['content'] * self.model_scores['content']
                active_models.append('content')
            except Exception as e:
                logging.warning(f"Content scoring failed: {e}")
        
        # Popularity scores
        if self.popularity.trained and self.weights.get('popularity', 0) > 0:
            try:
                pop_raw = [self.popularity.predict(aid) for aid in article_ids]
                self.model_scores['popularity'] = self._normalize_scores(pop_raw)
                final_scores += self.weights['popularity'] * self.model_scores['popularity']
                active_models.append('popularity')
            except Exception as e:
                logging.warning(f"Popularity scoring failed: {e}")
        
        if len(active_models) < len([k for k, v in self.weights.items() if v > 0]):
            logging.warning(f"Only {len(active_models)}/{len([k for k, v in self.weights.items() if v > 0])} models active: {active_models}")
        
        return final_scores.tolist()
    
    def rerank(self, user_id: str, article_ids: List[str]) -> List[str]:
        """Rerank articles using ensemble scores"""
        scores = self.score(user_id, article_ids)
        ranked_indices = np.argsort(scores)[::-1]
        return [article_ids[i] for i in ranked_indices]
    
    def set_weights(self, weights: Dict[str, float]):
        """Update model weights for A/B testing"""
        self.weights.update(weights)
        # Normalize weights to sum to 1
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}
    
    def save(self, path: str = "data/processed/ensemble_model.pkl"):
        """Save the entire ensemble"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'gnn': self.gnn,
                'ltr': self.ltr,
                'cf': self.cf,
                'content': self.content,
                'popularity': self.popularity,
                'weights': self.weights,
                'trained': self.trained
            }, f)
        logging.info(f"Ensemble saved to {path}")
    
    @classmethod
    def load(cls, path: str = "data/processed/ensemble_model.pkl",
             user_features_path: str = "data/processed/user_features.json",
             article_features_path: str = "data/processed/article_features.json") -> 'EnsembleReranker':
        """Load a saved ensemble"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        obj = cls(weights=data['weights'])
        obj.gnn = data['gnn']
        obj.ltr = data['ltr']
        obj.cf = data['cf']
        obj.content = data['content']
        obj.popularity = data['popularity']
        obj.trained = data['trained']
        
        # Load LTR features
        try:
            obj.ltr.load_features(user_features_path, article_features_path)
        except:
            pass
        
        return obj


# Keep HybridReranker for backward compatibility
class HybridReranker:
    """
    Legacy hybrid combining GNN and LTR scores
    Use EnsembleReranker for better results
    """
    
    def __init__(self, gnn_weight: float = 0.4, ltr_weight: float = 0.6):
        self.gnn = GNNRanker()
        self.ltr = LTRRanker()
        self.gnn_weight = gnn_weight
        self.ltr_weight = ltr_weight
        self.trained = False
        
    def train(self, interactions: List[Dict],
              graph_path: str = "data/processed/interaction_graph.pkl",
              user_features_path: str = "data/processed/user_features.json",
              article_features_path: str = "data/processed/article_features.json"):
        """Train both GNN and LTR models"""
        print("Training GNN model...")
        self.gnn.train(graph_path)
        
        print("Training LTR model...")
        self.ltr.load_features(user_features_path, article_features_path)
        self.ltr.train(interactions, user_features_path, article_features_path)
        
        self.trained = True
        return self
    
    def score(self, user_id: str, article_ids: List[str]) -> List[float]:
        """
        Combine GNN and LTR scores
        
        Returns:
            List of hybrid scores
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get scores from both models
        gnn_scores = np.array(self.gnn.score(user_id, article_ids))
        ltr_scores = np.array(self.ltr.score(user_id, article_ids))
        
        # Normalize scores to [0, 1]
        if gnn_scores.max() != gnn_scores.min():
            gnn_scores = (gnn_scores - gnn_scores.min()) / (gnn_scores.max() - gnn_scores.min())
        else:
            gnn_scores = np.ones_like(gnn_scores) * 0.5
            
        if ltr_scores.max() != ltr_scores.min():
            ltr_scores = (ltr_scores - ltr_scores.min()) / (ltr_scores.max() - ltr_scores.min())
        else:
            ltr_scores = np.ones_like(ltr_scores) * 0.5
        
        # Weighted combination
        hybrid_scores = self.gnn_weight * gnn_scores + self.ltr_weight * ltr_scores
        
        return hybrid_scores.tolist()
    
    def rerank(self, user_id: str, article_ids: List[str]) -> List[str]:
        """Rerank articles using hybrid scores"""
        scores = self.score(user_id, article_ids)
        ranked_indices = np.argsort(scores)[::-1]
        return [article_ids[i] for i in ranked_indices]
    
    def save(self, gnn_path: str = "data/processed/gnn_model.pkl",
             ltr_path: str = "data/processed/ltr_model.pkl"):
        """Save both models"""
        self.gnn.save(gnn_path)
        self.ltr.save(ltr_path)
    
    def load(self, gnn_path: str = "data/processed/gnn_model.pkl",
             ltr_path: str = "data/processed/ltr_model.pkl",
             user_features_path: str = "data/processed/user_features.json",
             article_features_path: str = "data/processed/article_features.json"):
        """Load both models"""
        self.gnn.load(gnn_path)
        self.ltr.load(ltr_path)
        self.ltr.load_features(user_features_path, article_features_path)
        self.trained = True
        return self


class PositionAwareReranker:
    """
    Smart reranker that respects position bias from ES results.
    Key insight: ES results are already relevant, personalization should be a boost, not a replacement.
    """
    
    def __init__(self, personalization_strength: float = 0.4):
        """
        Args:
            personalization_strength: How much to weight personalization vs position (0=pure position, 1=pure personalization)
        """
        self.personalization_strength = personalization_strength
        self.cf = None
        self.content = None
        self.popularity = None
        self.trained = False
        
    def train(self, interactions: List[Dict]):
        """Train the underlying models"""
        logging.info("Training Position-Aware Reranker...")
        
        # Use CF as primary personalization signal
        self.cf = CollaborativeFilteringEnsemble(user_cf_weight=0.4, item_cf_weight=0.3, mf_weight=0.3)
        self.cf.train(interactions)
        
        # Content for cold-start
        self.content = ContentBasedFilter()
        self.content.train(interactions)
        
        # Popularity for fallback
        self.popularity = PopularityRanker()
        self.popularity.train(interactions)
        
        self.trained = True
        logging.info("Position-Aware Reranker trained!")
        return self
    
    def _get_personalization_score(self, user_id: str, article_id: str) -> float:
        """Get combined personalization score"""
        cf_score = self.cf.predict(user_id, article_id) if self.cf and self.cf.trained else 0
        content_score = self.content.predict(user_id, article_id) if self.content and self.content.trained else 0
        pop_score = self.popularity.predict(article_id) if self.popularity and self.popularity.trained else 0
        
        # Weighted combination
        return 0.5 * cf_score + 0.3 * content_score + 0.2 * pop_score
    
    def rerank(self, user_id: str, article_ids: List[str]) -> List[str]:
        """
        Rerank with position bias awareness.
        Higher positions get a bonus, personalization provides the boost.
        """
        if not self.trained:
            return article_ids
        
        n = len(article_ids)
        scores = []
        
        for i, article_id in enumerate(article_ids):
            # Position score: exponential decay (position 1 gets 1.0, position 10 gets ~0.35)
            position_score = np.exp(-i * 0.1)
            
            # Personalization score
            personal_score = self._get_personalization_score(user_id, article_id)
            
            # Combine: position provides baseline, personalization boosts
            final_score = (1 - self.personalization_strength) * position_score + \
                          self.personalization_strength * (position_score + personal_score)
            
            scores.append((article_id, final_score))
        
        # Sort by final score
        scores.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scores]
    
    def save(self, path: str = "data/processed/position_aware_model.pkl"):
        """Save the model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'cf': self.cf,
                'content': self.content,
                'popularity': self.popularity,
                'personalization_strength': self.personalization_strength,
                'trained': self.trained
            }, f)
        logging.info(f"Position-Aware Reranker saved to {path}")
    
    @classmethod
    def load(cls, path: str = "data/processed/position_aware_model.pkl") -> 'PositionAwareReranker':
        """Load a saved model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        obj = cls(personalization_strength=data.get('personalization_strength', 0.4))
        obj.cf = data['cf']
        obj.content = data['content']
        obj.popularity = data['popularity']
        obj.trained = data['trained']
        return obj
