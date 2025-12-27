"""
Graph Neural Network for personalized ranking
Uses user-article-topic interaction graph to learn embeddings
Enhanced with attention mechanism and multi-hop propagation
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)


class GNNRanker:
    """
    Enhanced GNN-style ranker with attention-based aggregation
    Uses the interaction graph to propagate user preferences through topics
    """
    
    def __init__(self, embedding_dim: int = 64, num_iterations: int = 3, 
                 use_attention: bool = True, alpha: float = 0.6):
        self.embedding_dim = embedding_dim
        self.num_iterations = num_iterations
        self.use_attention = use_attention
        self.alpha = alpha  # Weight for self vs neighbor in aggregation
        
        self.user_embeddings = {}
        self.article_embeddings = {}
        self.topic_embeddings = {}
        self.graph = None
        
        # Attention weights (learned during training)
        self.attention_weights = {}
        
        # Store edge weights for scoring
        self.user_article_weights = {}  # (user, article) -> weight
        self.user_topic_weights = {}    # (user, topic) -> weight
        
        self.trained = False
        
    def load_graph(self, graph_path: str = "data/processed/interaction_graph.pkl"):
        """Load the interaction graph"""
        with open(graph_path, 'rb') as f:
            graph_builder = pickle.load(f)
        self.graph = graph_builder.graph
        self.user_nodes = graph_builder.user_nodes
        self.article_nodes = graph_builder.article_nodes
        self.topic_nodes = graph_builder.topic_nodes
        
        # Extract edge weights for scoring
        for user in self.user_nodes:
            for neighbor in self.graph.successors(user):
                weight = self.graph[user][neighbor].get('weight', 1.0)
                if neighbor in self.article_nodes:
                    self.user_article_weights[(user, neighbor)] = weight
                elif neighbor in self.topic_nodes:
                    self.user_topic_weights[(user, neighbor)] = weight
        
        return self
    
    def train(self, graph_path: str = "data/processed/interaction_graph.pkl"):
        """
        Train embeddings using enhanced graph propagation
        With attention-based neighbor aggregation
        """
        logging.info("Training Enhanced GNN Ranker...")
        self.load_graph(graph_path)
        
        # Initialize embeddings with Xavier initialization
        np.random.seed(42)
        scale = np.sqrt(2.0 / self.embedding_dim)
        
        for node in self.user_nodes:
            self.user_embeddings[node] = np.random.randn(self.embedding_dim) * scale
        for node in self.article_nodes:
            self.article_embeddings[node] = np.random.randn(self.embedding_dim) * scale
        for node in self.topic_nodes:
            self.topic_embeddings[node] = np.random.randn(self.embedding_dim) * scale
        
        # Initialize attention projection
        self.attention_proj = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.1
        
        # Propagate embeddings through graph
        logging.info(f"Running {self.num_iterations} propagation iterations...")
        for iteration in range(self.num_iterations):
            self._propagate_embeddings(iteration)
        
        # L2 Normalize embeddings
        for emb_dict in [self.user_embeddings, self.article_embeddings, self.topic_embeddings]:
            for key in emb_dict:
                norm = np.linalg.norm(emb_dict[key])
                if norm > 0:
                    emb_dict[key] = emb_dict[key] / norm
        
        self.trained = True
        logging.info(f"GNN trained: {len(self.user_embeddings)} users, "
                    f"{len(self.article_embeddings)} articles, {len(self.topic_embeddings)} topics")
        return self
    
    def _compute_attention(self, source_emb: np.ndarray, target_emb: np.ndarray, 
                            edge_weight: float = 1.0) -> float:
        """Compute attention coefficient between source and target embeddings"""
        if not self.use_attention:
            return edge_weight
        
        # Simple dot-product attention with edge weight
        proj_source = np.dot(source_emb, self.attention_proj)
        attention = np.dot(proj_source, target_emb)
        attention = 1 / (1 + np.exp(-attention))  # Sigmoid
        return attention * edge_weight
    
    def _propagate_embeddings(self, iteration: int):
        """Enhanced message passing with attention"""
        new_user_emb = {}
        new_article_emb = {}
        new_topic_emb = {}
        
        # Decay alpha over iterations for stability
        current_alpha = self.alpha * (0.9 ** iteration)
        
        # Update user embeddings from articles and topics
        for user in self.user_nodes:
            neighbors = list(self.graph.successors(user))
            if neighbors:
                neighbor_embs = []
                attention_sum = 0.0
                
                for neighbor in neighbors:
                    edge_weight = self.graph[user][neighbor].get('weight', 1.0)
                    
                    if neighbor in self.article_embeddings:
                        target_emb = self.article_embeddings[neighbor]
                    elif neighbor in self.topic_embeddings:
                        target_emb = self.topic_embeddings[neighbor]
                    else:
                        continue
                    
                    attention = self._compute_attention(
                        self.user_embeddings[user], target_emb, edge_weight
                    )
                    neighbor_embs.append(attention * target_emb)
                    attention_sum += attention
                
                if neighbor_embs and attention_sum > 0:
                    aggregated = np.sum(neighbor_embs, axis=0) / attention_sum
                    new_user_emb[user] = current_alpha * self.user_embeddings[user] + (1 - current_alpha) * aggregated
                else:
                    new_user_emb[user] = self.user_embeddings[user]
            else:
                new_user_emb[user] = self.user_embeddings[user]
        
        # Update article embeddings from topics AND users who clicked
        for article in self.article_nodes:
            neighbor_embs = []
            attention_sum = 0.0
            
            # From topics
            for neighbor in self.graph.successors(article):
                if neighbor in self.topic_embeddings:
                    target_emb = self.topic_embeddings[neighbor]
                    attention = self._compute_attention(
                        self.article_embeddings[article], target_emb, 1.0
                    )
                    neighbor_embs.append(attention * target_emb)
                    attention_sum += attention
            
            # From users (reverse edges - users who interacted)
            for user in self.graph.predecessors(article):
                if user in self.user_embeddings:
                    edge_weight = self.graph[user][article].get('weight', 1.0)
                    target_emb = self.user_embeddings[user]
                    attention = self._compute_attention(
                        self.article_embeddings[article], target_emb, edge_weight * 0.5
                    )
                    neighbor_embs.append(attention * target_emb)
                    attention_sum += attention
            
            if neighbor_embs and attention_sum > 0:
                aggregated = np.sum(neighbor_embs, axis=0) / attention_sum
                new_article_emb[article] = current_alpha * self.article_embeddings[article] + (1 - current_alpha) * aggregated
            else:
                new_article_emb[article] = self.article_embeddings[article]
        
        # Update topic embeddings from users who are interested
        for topic in self.topic_nodes:
            neighbor_embs = []
            attention_sum = 0.0
            
            for pred in self.graph.predecessors(topic):
                if pred in self.user_embeddings:
                    edge_weight = self.graph[pred][topic].get('weight', 1.0)
                    target_emb = self.user_embeddings[pred]
                    attention = self._compute_attention(
                        self.topic_embeddings[topic], target_emb, edge_weight
                    )
                    neighbor_embs.append(attention * target_emb)
                    attention_sum += attention
            
            if neighbor_embs and attention_sum > 0:
                aggregated = np.sum(neighbor_embs, axis=0) / attention_sum
                new_topic_emb[topic] = current_alpha * self.topic_embeddings[topic] + (1 - current_alpha) * aggregated
            else:
                new_topic_emb[topic] = self.topic_embeddings[topic]
        
        self.user_embeddings = new_user_emb
        self.article_embeddings = new_article_emb
        self.topic_embeddings = new_topic_emb
    
    def score(self, user_id: str, article_ids: List[str]) -> List[float]:
        """
        Score articles for a user based on embedding similarity + edge weights
        
        Returns:
            List of scores (higher = more relevant)
        """
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get user embedding
        if user_id in self.user_embeddings:
            user_emb = self.user_embeddings[user_id]
            is_cold_user = False
        else:
            # Cold start: use average embedding as fallback
            is_cold_user = True
            if self.user_embeddings:
                user_emb = np.mean(list(self.user_embeddings.values()), axis=0)
            else:
                user_emb = np.zeros(self.embedding_dim)
        
        # Pre-compute user's topic preferences for efficiency
        user_topic_prefs = {}
        for (uid, topic), weight in self.user_topic_weights.items():
            if uid == user_id and topic in self.topic_embeddings:
                user_topic_prefs[topic] = (self.topic_embeddings[topic], weight)
        
        scores = []
        for article_id in article_ids:
            score = 0.0
            
            if article_id in self.article_embeddings:
                article_emb = self.article_embeddings[article_id]
                
                # Cosine similarity (embeddings are normalized, so dot product = cosine)
                embedding_score = np.dot(user_emb, article_emb)
                
                # Shift to positive range [0, 2] since cosine is [-1, 1]
                embedding_score = (embedding_score + 1) / 2
                
                if not is_cold_user:
                    # Direct edge weight bonus (user clicked this before)
                    edge_weight = self.user_article_weights.get((user_id, article_id), 0.0)
                    
                    # Topic overlap bonus
                    topic_bonus = 0.0
                    for topic, (topic_emb, topic_weight) in user_topic_prefs.items():
                        topic_sim = np.dot(article_emb, topic_emb)
                        topic_sim = (topic_sim + 1) / 2  # Normalize to [0, 1]
                        topic_bonus += topic_sim * topic_weight * 0.3
                    
                    score = embedding_score + 0.5 * edge_weight + topic_bonus
                else:
                    # Cold user: just use embedding similarity
                    score = embedding_score * 0.5  # Reduce confidence for cold users
            else:
                # Article not in graph - minimal score
                score = 0.1  # Small non-zero to not disrupt ranking completely
            
            scores.append(score)
        
        return scores
    
    def rerank(self, user_id: str, article_ids: List[str], 
               use_position_blend: bool = True) -> List[str]:
        """
        Rerank articles for a user
        
        Args:
            user_id: User ID
            article_ids: Article IDs in ES order (most relevant first)
            use_position_blend: If True, blend with ES position for query relevance
        """
        scores = self.score(user_id, article_ids)
        
        # Check if all scores are 0 (cold user or all cold articles)
        if all(s == 0 for s in scores):
            # Return original order (baseline)
            return article_ids
        
        # Normalize GNN scores to [0, 1]
        if max(scores) > min(scores):
            gnn_scores = [(s - min(scores)) / (max(scores) - min(scores)) for s in scores]
        else:
            gnn_scores = [0.5] * len(scores)
        
        if use_position_blend:
            # Blend with ES position score to maintain query relevance
            final_scores = []
            for i, (article_id, gnn_score) in enumerate(zip(article_ids, gnn_scores)):
                # ES position score (gentle decay)
                es_score = 1.0 / (1 + i * 0.1)  # Position 1=1.0, 10=0.5
                
                # Blend: 60% GNN (personalization) + 40% ES (query relevance)
                final = 0.6 * gnn_score + 0.4 * es_score
                final_scores.append((article_id, final))
            
            final_scores.sort(key=lambda x: x[1], reverse=True)
            return [aid for aid, _ in final_scores]
        else:
            ranked_indices = np.argsort(scores)[::-1]  # Descending order
            return [article_ids[i] for i in ranked_indices]
    
    def get_user_coverage(self, user_id: str) -> float:
        """Check how well we can score for this user (0=cold, 1=full)"""
        if user_id not in self.user_embeddings:
            return 0.0
        if user_id not in [u for u, _ in self.user_article_weights.keys()]:
            return 0.3  # Has embedding but no direct edges
        return 1.0
    
    def save(self, path: str = "data/processed/gnn_model.pkl"):
        """Save trained model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'user_embeddings': self.user_embeddings,
                'article_embeddings': self.article_embeddings,
                'topic_embeddings': self.topic_embeddings,
                'embedding_dim': self.embedding_dim,
                'num_iterations': self.num_iterations,
                'use_attention': self.use_attention,
                'alpha': self.alpha,
                'attention_proj': self.attention_proj if hasattr(self, 'attention_proj') else None,
                'user_article_weights': self.user_article_weights,
                'user_topic_weights': self.user_topic_weights,
                'trained': self.trained
            }, f)
        logging.info(f"GNN model saved to {path}")
    
    def load(self, path: str = "data/processed/gnn_model.pkl"):
        """Load trained model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.user_embeddings = data['user_embeddings']
        self.article_embeddings = data['article_embeddings']
        self.topic_embeddings = data['topic_embeddings']
        self.embedding_dim = data['embedding_dim']
        self.num_iterations = data.get('num_iterations', 3)
        self.use_attention = data.get('use_attention', True)
        self.alpha = data.get('alpha', 0.6)
        self.attention_proj = data.get('attention_proj')
        self.user_article_weights = data.get('user_article_weights', {})
        self.user_topic_weights = data.get('user_topic_weights', {})
        self.trained = data['trained']
        return self
