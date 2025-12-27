"""
Learning to Rank model for personalized ranking
Uses user and article features to predict click probability
"""
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class LTRRanker:
    """
    Learning to Rank using gradient boosted trees
    Falls back to simple logistic-style scoring if sklearn not available
    """
    
    def __init__(self):
        self.user_features = {}
        self.article_features = {}
        self.feature_weights = None
        self.trained = False
        self.use_xgboost = False
        self.model = None
        
    def load_features(self, 
                     user_features_path: str = "data/processed/user_features.json",
                     article_features_path: str = "data/processed/article_features.json"):
        """Load pre-computed features"""
        with open(user_features_path, 'r') as f:
            self.user_features = json.load(f)
        with open(article_features_path, 'r') as f:
            self.article_features = json.load(f)
        return self
    
    def _get_feature_vector(self, user_id: str, article_id: str) -> np.ndarray:
        """Create feature vector for user-article pair"""
        features = []
        
        # User features
        user = self.user_features.get(user_id, {})
        features.append(user.get('overall_ctr', 0.0))
        features.append(user.get('avg_dwell_time', 0.0) / 60.0)  # Normalize
        features.append(user.get('topic_diversity', 0) / 10.0)
        features.append(user.get('total_queries', 0) / 100.0)
        
        # Article features
        article = self.article_features.get(article_id, {})
        features.append(article.get('historical_ctr', 0.0))
        features.append(article.get('engagement_rate', 0.0))
        features.append(article.get('word_count', 0) / 1000.0)
        features.append(article.get('num_topics', 0) / 5.0)
        
        # Cross features: topic affinity
        user_affinities = user.get('topic_affinities', {})
        article_topics = article.get('topics', [])
        
        max_affinity = 0.0
        avg_affinity = 0.0
        if article_topics and user_affinities:
            affinities = [
                user_affinities.get(topic, {}).get('affinity_score', 0.0)
                for topic in article_topics
            ]
            if affinities:
                max_affinity = max(affinities)
                avg_affinity = np.mean(affinities)
        
        features.append(max_affinity)
        features.append(avg_affinity)
        
        # Topic match indicator
        user_top_topics = set(user.get('top_topics', [])[:3])
        topic_match = len(user_top_topics.intersection(set(article_topics))) / max(len(user_top_topics), 1)
        features.append(topic_match)
        
        return np.array(features, dtype=np.float32)
    
    def train(self, interactions: List[Dict], 
              user_features_path: str = "data/processed/user_features.json",
              article_features_path: str = "data/processed/article_features.json"):
        """
        Train the LTR model on interaction data
        """
        self.load_features(user_features_path, article_features_path)
        
        # Prepare training data
        X = []
        y = []
        
        for interaction in interactions:
            user_id = interaction['user_id']
            for action in interaction['actions']:
                article_id = action['article_id']
                features = self._get_feature_vector(user_id, article_id)
                X.append(features)
                
                # Label: 1 if clicked, 0 otherwise
                # Could also use graded relevance
                if action.get('liked') or action.get('shared') or action.get('bookmarked'):
                    label = 1.0
                elif action.get('clicked') and action.get('dwell_time_secs', 0) > 10:
                    label = 0.8
                elif action.get('clicked'):
                    label = 0.5
                else:
                    label = 0.0
                y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Try to use sklearn, otherwise use simple linear model
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            # Convert to binary for classifier
            y_binary = (y > 0.3).astype(int)
            self.model.fit(X, y_binary)
            self.use_xgboost = True
        except ImportError:
            # Fallback: learn feature weights using simple correlation
            self.feature_weights = np.zeros(X.shape[1])
            for i in range(X.shape[1]):
                if np.std(X[:, i]) > 0:
                    corr = np.corrcoef(X[:, i], y)[0, 1]
                    if not np.isnan(corr):
                        self.feature_weights[i] = corr
            self.use_xgboost = False
        
        self.trained = True
        return self
    
    def score(self, user_id: str, article_ids: List[str]) -> List[float]:
        """Score articles for a user"""
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        scores = []
        for article_id in article_ids:
            features = self._get_feature_vector(user_id, article_id)
            
            if self.use_xgboost and self.model is not None:
                # Get probability of positive class
                prob = self.model.predict_proba(features.reshape(1, -1))[0, 1]
                scores.append(prob)
            else:
                # Simple weighted sum
                score = np.dot(features, self.feature_weights)
                scores.append(score)
        
        return scores
    
    def rerank(self, user_id: str, article_ids: List[str]) -> List[str]:
        """Rerank articles for a user"""
        scores = self.score(user_id, article_ids)
        ranked_indices = np.argsort(scores)[::-1]
        return [article_ids[i] for i in ranked_indices]
    
    def save(self, path: str = "data/processed/ltr_model.pkl"):
        """Save trained model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_weights': self.feature_weights,
                'use_xgboost': self.use_xgboost,
                'trained': self.trained
            }, f)
    
    def load(self, path: str = "data/processed/ltr_model.pkl"):
        """Load trained model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_weights = data['feature_weights']
        self.use_xgboost = data['use_xgboost']
        self.trained = data['trained']
        return self
