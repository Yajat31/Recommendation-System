"""
Collaborative Filtering for personalized news ranking
Implements User-Based CF, Item-Based CF, and Matrix Factorization
"""
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)


class UserBasedCF:
    """
    User-Based Collaborative Filtering
    Finds similar users based on interaction patterns and recommends articles
    """
    
    def __init__(self, k_neighbors: int = 10, min_common: int = 2):
        self.k_neighbors = k_neighbors
        self.min_common = min_common
        self.user_article_matrix = {}  # user_id -> {article_id: score}
        self.user_similarities = {}    # user_id -> [(similar_user, similarity)]
        self.article_to_users = {}     # article_id -> set(user_ids)
        self.trained = False
        
    def train(self, interactions: List[Dict]):
        """Build user-item matrix and compute similarities"""
        logging.info("Training User-Based CF...")
        
        # Build user-article interaction matrix
        for interaction in interactions:
            user_id = interaction.get('user_id', '')
            if not user_id:
                continue
                
            if user_id not in self.user_article_matrix:
                self.user_article_matrix[user_id] = {}
            
            actions = interaction.get('actions', [])
            for action in actions:
                article_id = action.get('article_id', '')
                if not article_id:
                    continue
                
                # Compute engagement score
                score = self._compute_engagement_score(action)
                if score > 0:
                    self.user_article_matrix[user_id][article_id] = max(
                        self.user_article_matrix[user_id].get(article_id, 0), 
                        score
                    )
                    
                    # Track which users interacted with each article
                    if article_id not in self.article_to_users:
                        self.article_to_users[article_id] = set()
                    self.article_to_users[article_id].add(user_id)
        
        # Compute user similarities
        users = list(self.user_article_matrix.keys())
        logging.info(f"Computing similarities for {len(users)} users...")
        
        for user in users:
            similarities = []
            user_articles = set(self.user_article_matrix[user].keys())
            
            for other_user in users:
                if user == other_user:
                    continue
                    
                other_articles = set(self.user_article_matrix[other_user].keys())
                common = user_articles & other_articles
                
                if len(common) >= self.min_common:
                    sim = self._cosine_similarity(user, other_user, common)
                    if sim > 0:
                        similarities.append((other_user, sim))
            
            # Keep top k neighbors
            similarities.sort(key=lambda x: x[1], reverse=True)
            self.user_similarities[user] = similarities[:self.k_neighbors]
        
        self.trained = True
        logging.info(f"User-Based CF trained: {len(users)} users, {len(self.article_to_users)} articles")
        return self
    
    def _compute_engagement_score(self, action: Dict) -> float:
        """Compute engagement score from action"""
        score = 0.0
        if action.get('clicked', False):
            score += 1.0
        dwell = action.get('dwell_time_secs', 0)
        if dwell > 0:
            score += min(dwell / 30.0, 2.0)  # Cap at 2 for 30+ seconds
        if action.get('liked', False):
            score += 3.0
        if action.get('bookmarked', False):
            score += 2.0
        if action.get('shared', False):
            score += 4.0
        return score
    
    def _cosine_similarity(self, user1: str, user2: str, common_articles: set) -> float:
        """Compute cosine similarity between two users"""
        if not common_articles:
            return 0.0
            
        vec1 = [self.user_article_matrix[user1].get(a, 0) for a in common_articles]
        vec2 = [self.user_article_matrix[user2].get(a, 0) for a in common_articles]
        
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = np.sqrt(sum(a * a for a in vec1))
        norm2 = np.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
    
    def predict(self, user_id: str, article_id: str) -> float:
        """Predict user interest in article"""
        if not self.trained:
            return 0.0
            
        if user_id not in self.user_similarities:
            # Cold start: return popularity score
            return len(self.article_to_users.get(article_id, set())) / max(len(self.user_article_matrix), 1)
        
        # Weighted average of similar users' ratings
        similar_users = self.user_similarities.get(user_id, [])
        if not similar_users:
            return 0.0
            
        numerator = 0.0
        denominator = 0.0
        
        for other_user, similarity in similar_users:
            if article_id in self.user_article_matrix.get(other_user, {}):
                numerator += similarity * self.user_article_matrix[other_user][article_id]
                denominator += similarity
        
        if denominator == 0:
            return 0.0
        return numerator / denominator
    
    def rerank(self, user_id: str, article_ids: List[str]) -> List[str]:
        """Rerank articles for a user"""
        scores = [(aid, self.predict(user_id, aid)) for aid in article_ids]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scores]


class ItemBasedCF:
    """
    Item-Based Collaborative Filtering
    Finds similar articles based on user interaction patterns
    """
    
    def __init__(self, k_neighbors: int = 20, min_common: int = 2):
        self.k_neighbors = k_neighbors
        self.min_common = min_common
        self.article_user_matrix = {}  # article_id -> {user_id: score}
        self.article_similarities = {} # article_id -> [(similar_article, similarity)]
        self.user_history = {}         # user_id -> {article_id: score}
        self.trained = False
        
    def train(self, interactions: List[Dict]):
        """Build article-user matrix and compute similarities"""
        logging.info("Training Item-Based CF...")
        
        # Build article-user and user history matrices
        for interaction in interactions:
            user_id = interaction.get('user_id', '')
            if not user_id:
                continue
            
            if user_id not in self.user_history:
                self.user_history[user_id] = {}
            
            actions = interaction.get('actions', [])
            for action in actions:
                article_id = action.get('article_id', '')
                if not article_id:
                    continue
                
                score = self._compute_engagement_score(action)
                if score > 0:
                    if article_id not in self.article_user_matrix:
                        self.article_user_matrix[article_id] = {}
                    
                    self.article_user_matrix[article_id][user_id] = max(
                        self.article_user_matrix[article_id].get(user_id, 0),
                        score
                    )
                    self.user_history[user_id][article_id] = max(
                        self.user_history[user_id].get(article_id, 0),
                        score
                    )
        
        # Compute article similarities
        articles = list(self.article_user_matrix.keys())
        logging.info(f"Computing similarities for {len(articles)} articles...")
        
        for article in articles:
            similarities = []
            article_users = set(self.article_user_matrix[article].keys())
            
            for other_article in articles:
                if article == other_article:
                    continue
                
                other_users = set(self.article_user_matrix[other_article].keys())
                common = article_users & other_users
                
                if len(common) >= self.min_common:
                    sim = self._cosine_similarity(article, other_article, common)
                    if sim > 0:
                        similarities.append((other_article, sim))
            
            similarities.sort(key=lambda x: x[1], reverse=True)
            self.article_similarities[article] = similarities[:self.k_neighbors]
        
        self.trained = True
        logging.info(f"Item-Based CF trained: {len(articles)} articles, {len(self.user_history)} users")
        return self
    
    def _compute_engagement_score(self, action: Dict) -> float:
        """Compute engagement score from action"""
        score = 0.0
        if action.get('clicked', False):
            score += 1.0
        dwell = action.get('dwell_time_secs', 0)
        if dwell > 0:
            score += min(dwell / 30.0, 2.0)
        if action.get('liked', False):
            score += 3.0
        if action.get('bookmarked', False):
            score += 2.0
        if action.get('shared', False):
            score += 4.0
        return score
    
    def _cosine_similarity(self, article1: str, article2: str, common_users: set) -> float:
        """Compute cosine similarity between two articles"""
        if not common_users:
            return 0.0
            
        vec1 = [self.article_user_matrix[article1].get(u, 0) for u in common_users]
        vec2 = [self.article_user_matrix[article2].get(u, 0) for u in common_users]
        
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = np.sqrt(sum(a * a for a in vec1))
        norm2 = np.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
    
    def predict(self, user_id: str, article_id: str) -> float:
        """Predict user interest in article based on similar articles they liked"""
        if not self.trained:
            return 0.0
        
        user_articles = self.user_history.get(user_id, {})
        if not user_articles:
            # Cold start: return popularity
            return len(self.article_user_matrix.get(article_id, {})) / max(len(self.user_history), 1)
        
        # Find similar articles user has interacted with
        similar_articles = self.article_similarities.get(article_id, [])
        if not similar_articles:
            return 0.0
        
        numerator = 0.0
        denominator = 0.0
        
        for similar_article, similarity in similar_articles:
            if similar_article in user_articles:
                numerator += similarity * user_articles[similar_article]
                denominator += similarity
        
        if denominator == 0:
            return 0.0
        return numerator / denominator
    
    def get_similar_items(self, article_id: str, n: int = 10) -> List[Tuple[str, float]]:
        """Get n most similar items to the given article"""
        if not self.trained or article_id not in self.article_similarities:
            return []
        
        similar = self.article_similarities.get(article_id, [])
        return similar[:n]
    
    def rerank(self, user_id: str, article_ids: List[str]) -> List[str]:
        """Rerank articles for a user"""
        scores = [(aid, self.predict(user_id, aid)) for aid in article_ids]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scores]


class MatrixFactorization:
    """
    Matrix Factorization using Alternating Least Squares (ALS)
    Learns latent factors for users and articles
    """
    
    def __init__(self, n_factors: int = 32, n_iterations: int = 20, 
                 reg_lambda: float = 0.1, learning_rate: float = 0.01):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg_lambda = reg_lambda
        self.learning_rate = learning_rate
        
        self.user_factors = {}   # user_id -> np.array
        self.item_factors = {}   # article_id -> np.array
        self.user_bias = {}
        self.item_bias = {}
        self.global_bias = 0.0
        
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        
        self.trained = False
        
    def train(self, interactions: List[Dict]):
        """Train matrix factorization model using SGD"""
        logging.info("Training Matrix Factorization...")
        
        # Build user-item ratings
        ratings = []  # (user_id, article_id, rating)
        
        for interaction in interactions:
            user_id = interaction.get('user_id', '')
            if not user_id:
                continue
            
            actions = interaction.get('actions', [])
            for action in actions:
                article_id = action.get('article_id', '')
                if not article_id:
                    continue
                
                score = self._compute_engagement_score(action)
                if score > 0:
                    ratings.append((user_id, article_id, score))
        
        if not ratings:
            logging.warning("No ratings found for matrix factorization")
            return self
        
        # Create mappings
        users = list(set(r[0] for r in ratings))
        items = list(set(r[1] for r in ratings))
        
        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.item_to_idx = {a: i for i, a in enumerate(items)}
        self.idx_to_item = {i: a for a, i in self.item_to_idx.items()}
        
        n_users = len(users)
        n_items = len(items)
        
        # Initialize factors
        np.random.seed(42)
        U = np.random.randn(n_users, self.n_factors) * 0.1
        V = np.random.randn(n_items, self.n_factors) * 0.1
        bu = np.zeros(n_users)
        bi = np.zeros(n_items)
        
        # Global mean
        self.global_bias = np.mean([r[2] for r in ratings])
        
        # SGD training
        logging.info(f"Training with {len(ratings)} ratings, {n_users} users, {n_items} items")
        
        for iteration in range(self.n_iterations):
            np.random.shuffle(ratings)
            total_error = 0.0
            
            for user_id, item_id, rating in ratings:
                u_idx = self.user_to_idx[user_id]
                i_idx = self.item_to_idx[item_id]
                
                # Predict
                pred = self.global_bias + bu[u_idx] + bi[i_idx] + np.dot(U[u_idx], V[i_idx])
                error = rating - pred
                total_error += error ** 2
                
                # Update biases
                bu[u_idx] += self.learning_rate * (error - self.reg_lambda * bu[u_idx])
                bi[i_idx] += self.learning_rate * (error - self.reg_lambda * bi[i_idx])
                
                # Update factors
                U[u_idx] += self.learning_rate * (error * V[i_idx] - self.reg_lambda * U[u_idx])
                V[i_idx] += self.learning_rate * (error * U[u_idx] - self.reg_lambda * V[i_idx])
            
            if iteration % 5 == 0:
                rmse = np.sqrt(total_error / len(ratings))
                logging.info(f"Iteration {iteration}: RMSE = {rmse:.4f}")
        
        # Store factors
        for user_id, idx in self.user_to_idx.items():
            self.user_factors[user_id] = U[idx]
            self.user_bias[user_id] = bu[idx]
            
        for item_id, idx in self.item_to_idx.items():
            self.item_factors[item_id] = V[idx]
            self.item_bias[item_id] = bi[idx]
        
        self.trained = True
        logging.info(f"Matrix Factorization trained: {n_users} users, {n_items} items, {self.n_factors} factors")
        return self
    
    def _compute_engagement_score(self, action: Dict) -> float:
        """Compute engagement score from action"""
        score = 0.0
        if action.get('clicked', False):
            score += 1.0
        dwell = action.get('dwell_time_secs', 0)
        if dwell > 0:
            score += min(dwell / 30.0, 2.0)
        if action.get('liked', False):
            score += 3.0
        if action.get('bookmarked', False):
            score += 2.0
        if action.get('shared', False):
            score += 4.0
        return score
    
    def predict(self, user_id: str, article_id: str) -> float:
        """Predict user-item rating"""
        if not self.trained:
            return self.global_bias
        
        user_factor = self.user_factors.get(user_id)
        item_factor = self.item_factors.get(article_id)
        
        if user_factor is None or item_factor is None:
            # Cold start
            return self.global_bias
        
        pred = (self.global_bias + 
                self.user_bias.get(user_id, 0) + 
                self.item_bias.get(article_id, 0) + 
                np.dot(user_factor, item_factor))
        
        return max(0, pred)  # Ensure non-negative
    
    def rerank(self, user_id: str, article_ids: List[str]) -> List[str]:
        """Rerank articles for a user"""
        scores = [(aid, self.predict(user_id, aid)) for aid in article_ids]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scores]


class CollaborativeFilteringEnsemble:
    """
    Ensemble of collaborative filtering methods
    Combines User-CF, Item-CF, and Matrix Factorization
    """
    
    def __init__(self, user_cf_weight: float = 0.3, 
                 item_cf_weight: float = 0.3, 
                 mf_weight: float = 0.4):
        self.user_cf = UserBasedCF(k_neighbors=10, min_common=1)
        self.item_cf = ItemBasedCF(k_neighbors=20, min_common=1)
        self.mf = MatrixFactorization(n_factors=32, n_iterations=20)
        
        self.user_cf_weight = user_cf_weight
        self.item_cf_weight = item_cf_weight
        self.mf_weight = mf_weight
        
        self.trained = False
    
    def train(self, interactions: List[Dict]):
        """Train all CF models"""
        logging.info("=" * 50)
        logging.info("Training Collaborative Filtering Ensemble")
        logging.info("=" * 50)
        
        self.user_cf.train(interactions)
        self.item_cf.train(interactions)
        self.mf.train(interactions)
        
        self.trained = True
        logging.info("Ensemble training complete!")
        return self
    
    def predict(self, user_id: str, article_id: str) -> float:
        """Combined prediction from all models"""
        if not self.trained:
            return 0.0
        
        user_cf_score = self.user_cf.predict(user_id, article_id)
        item_cf_score = self.item_cf.predict(user_id, article_id)
        mf_score = self.mf.predict(user_id, article_id)
        
        # Normalize each score independently to [0, 1] using model-specific ranges
        # User-CF scores are typically in [0, max_engagement] range
        user_cf_norm = min(user_cf_score / 5.0, 1.0) if user_cf_score > 0 else 0
        
        # Item-CF scores similar range
        item_cf_norm = min(item_cf_score / 5.0, 1.0) if item_cf_score > 0 else 0
        
        # MF score is typically around global_bias (1-2 range)
        mf_norm = min(mf_score / 3.0, 1.0) if mf_score > 0 else 0
        
        combined = (self.user_cf_weight * user_cf_norm + 
                   self.item_cf_weight * item_cf_norm + 
                   self.mf_weight * mf_norm)
        
        return combined
    
    def rerank(self, user_id: str, article_ids: List[str]) -> List[str]:
        """Rerank articles using ensemble"""
        scores = [(aid, self.predict(user_id, aid)) for aid in article_ids]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scores]
    
    def save(self, path: str = "data/processed/cf_model.pkl"):
        """Save the ensemble model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logging.info(f"CF Ensemble saved to {path}")
    
    @classmethod
    def load(cls, path: str = "data/processed/cf_model.pkl") -> 'CollaborativeFilteringEnsemble':
        """Load a saved model"""
        with open(path, 'rb') as f:
            return pickle.load(f)
