"""
Content-Based Filtering for personalized news ranking
Uses TF-IDF and topic similarity for recommendations
"""
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import logging
import re

logging.basicConfig(level=logging.INFO)


class ContentBasedFilter:
    """
    Content-Based Filtering using TF-IDF and topic matching
    Recommends articles similar to user's historical preferences
    """
    
    def __init__(self, top_k_terms: int = 100):
        self.top_k_terms = top_k_terms
        
        # Article representations
        self.article_tfidf = {}       # article_id -> {term: tfidf_score}
        self.article_topics = {}      # article_id -> [topics]
        self.article_vectors = {}     # article_id -> normalized vector
        
        # User profiles
        self.user_topic_prefs = {}    # user_id -> {topic: score}
        self.user_term_prefs = {}     # user_id -> {term: score}
        self.user_vectors = {}        # user_id -> normalized vector
        
        # IDF values
        self.idf = {}
        self.vocabulary = set()
        
        self.trained = False
        
    def train(self, interactions: List[Dict], article_features_path: str = "data/processed/article_features.json"):
        """Train content-based model"""
        logging.info("Training Content-Based Filter...")
        
        # Load article features if available
        try:
            with open(article_features_path, 'r') as f:
                article_features = json.load(f)
        except FileNotFoundError:
            article_features = {}
        
        # Build article representations from interactions
        article_docs = {}  # article_id -> text
        
        for interaction in interactions:
            actions = interaction.get('actions', [])
            for action in actions:
                article_id = action.get('article_id', '')
                if not article_id:
                    continue
                
                text = action.get('text_preview', '')
                topics = action.get('topics', [])
                
                if article_id not in article_docs:
                    article_docs[article_id] = text
                    self.article_topics[article_id] = topics
        
        # Add from article features
        for article_id, features in article_features.items():
            if article_id not in self.article_topics:
                self.article_topics[article_id] = features.get('topics', [])
        
        # Compute TF-IDF
        logging.info(f"Computing TF-IDF for {len(article_docs)} articles...")
        self._compute_tfidf(article_docs)
        
        # Build user profiles from interactions
        logging.info("Building user profiles...")
        for interaction in interactions:
            user_id = interaction.get('user_id', '')
            if not user_id:
                continue
            
            if user_id not in self.user_topic_prefs:
                self.user_topic_prefs[user_id] = defaultdict(float)
                self.user_term_prefs[user_id] = defaultdict(float)
            
            actions = interaction.get('actions', [])
            for action in actions:
                article_id = action.get('article_id', '')
                engagement = self._compute_engagement_score(action)
                
                if engagement > 0:
                    # Update topic preferences
                    topics = action.get('topics', self.article_topics.get(article_id, []))
                    for topic in topics:
                        self.user_topic_prefs[user_id][topic] += engagement
                    
                    # Update term preferences
                    if article_id in self.article_tfidf:
                        for term, score in self.article_tfidf[article_id].items():
                            self.user_term_prefs[user_id][term] += engagement * score
        
        # Normalize user profiles
        for user_id in self.user_topic_prefs:
            total = sum(self.user_topic_prefs[user_id].values())
            if total > 0:
                for topic in self.user_topic_prefs[user_id]:
                    self.user_topic_prefs[user_id][topic] /= total
            
            # Keep top terms
            term_prefs = self.user_term_prefs[user_id]
            if term_prefs:
                top_terms = sorted(term_prefs.items(), key=lambda x: x[1], reverse=True)[:self.top_k_terms]
                self.user_term_prefs[user_id] = dict(top_terms)
                
                # Normalize
                total = sum(self.user_term_prefs[user_id].values())
                if total > 0:
                    for term in self.user_term_prefs[user_id]:
                        self.user_term_prefs[user_id][term] /= total
        
        self.trained = True
        logging.info(f"Content-Based Filter trained: {len(self.user_topic_prefs)} users, {len(self.article_tfidf)} articles")
        return self
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
                    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                    'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                    'this', 'that', 'these', 'those', 'it', 'its', 'as', 'if', 'then'}
        return [t for t in tokens if t not in stopwords and len(t) > 2]
    
    def _compute_tfidf(self, article_docs: Dict[str, str]):
        """Compute TF-IDF for all articles"""
        # Document frequency
        df = Counter()
        article_tf = {}
        
        for article_id, text in article_docs.items():
            tokens = self._tokenize(text)
            if not tokens:
                continue
            
            # Term frequency
            tf = Counter(tokens)
            total = len(tokens)
            article_tf[article_id] = {t: c / total for t, c in tf.items()}
            
            # Update document frequency
            for term in set(tokens):
                df[term] += 1
        
        # IDF
        n_docs = len(article_docs)
        self.idf = {term: np.log(n_docs / (count + 1)) for term, count in df.items()}
        self.vocabulary = set(self.idf.keys())
        
        # TF-IDF
        for article_id, tf in article_tf.items():
            tfidf = {}
            for term, freq in tf.items():
                tfidf[term] = freq * self.idf.get(term, 0)
            
            # Keep top terms
            top_terms = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:self.top_k_terms]
            self.article_tfidf[article_id] = dict(top_terms)
    
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
        """Predict user interest based on content similarity"""
        if not self.trained:
            return 0.0
        
        # Topic similarity
        topic_score = 0.0
        user_topics = self.user_topic_prefs.get(user_id, {})
        article_topics = self.article_topics.get(article_id, [])
        
        if user_topics and article_topics:
            for topic in article_topics:
                topic_score += user_topics.get(topic, 0)
            topic_score /= len(article_topics)
        
        # Term similarity
        term_score = 0.0
        user_terms = self.user_term_prefs.get(user_id, {})
        article_terms = self.article_tfidf.get(article_id, {})
        
        if user_terms and article_terms:
            common_terms = set(user_terms.keys()) & set(article_terms.keys())
            if common_terms:
                for term in common_terms:
                    term_score += user_terms[term] * article_terms[term]
        
        # Combined score (weighted)
        return 0.6 * topic_score + 0.4 * term_score
    
    def rerank(self, user_id: str, article_ids: List[str]) -> List[str]:
        """Rerank articles based on content similarity"""
        scores = [(aid, self.predict(user_id, aid)) for aid in article_ids]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scores]
    
    def save(self, path: str = "data/processed/content_model.pkl"):
        """Save the model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logging.info(f"Content-Based Filter saved to {path}")
    
    @classmethod
    def load(cls, path: str = "data/processed/content_model.pkl") -> 'ContentBasedFilter':
        """Load a saved model"""
        with open(path, 'rb') as f:
            return pickle.load(f)


class PopularityRanker:
    """
    Simple popularity-based ranker
    Useful as a baseline and for cold-start scenarios
    """
    
    def __init__(self, decay_factor: float = 0.95):
        self.decay_factor = decay_factor
        self.article_scores = {}
        self.article_click_counts = Counter()
        self.article_engagement_scores = defaultdict(float)
        self.trained = False
        
    def train(self, interactions: List[Dict]):
        """Compute article popularity scores"""
        logging.info("Training Popularity Ranker...")
        
        for interaction in interactions:
            actions = interaction.get('actions', [])
            for action in actions:
                article_id = action.get('article_id', '')
                if not article_id:
                    continue
                
                if action.get('clicked', False):
                    self.article_click_counts[article_id] += 1
                
                engagement = 0.0
                if action.get('clicked', False):
                    engagement += 1.0
                if action.get('liked', False):
                    engagement += 3.0
                if action.get('bookmarked', False):
                    engagement += 2.0
                if action.get('shared', False):
                    engagement += 4.0
                dwell = action.get('dwell_time_secs', 0)
                if dwell > 0:
                    engagement += min(dwell / 30.0, 2.0)
                
                self.article_engagement_scores[article_id] += engagement
        
        # Normalize scores
        max_engagement = max(self.article_engagement_scores.values()) if self.article_engagement_scores else 1
        for article_id in self.article_engagement_scores:
            self.article_scores[article_id] = self.article_engagement_scores[article_id] / max_engagement
        
        self.trained = True
        logging.info(f"Popularity Ranker trained: {len(self.article_scores)} articles")
        return self
    
    def predict(self, article_id: str) -> float:
        """Get popularity score for article"""
        return self.article_scores.get(article_id, 0.0)
    
    def rerank(self, article_ids: List[str]) -> List[str]:
        """Rerank by popularity"""
        scores = [(aid, self.predict(aid)) for aid in article_ids]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scores]
    
    def save(self, path: str = "data/processed/popularity_model.pkl"):
        """Save the model"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str = "data/processed/popularity_model.pkl") -> 'PopularityRanker':
        """Load a saved model"""
        with open(path, 'rb') as f:
            return pickle.load(f)
