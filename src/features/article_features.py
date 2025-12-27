"""
Article feature extraction
"""
import json
import numpy as np
from typing import Dict, List, Any
from collections import Counter
from pathlib import Path


class ArticleFeatureExtractor:
    """Extract article-level features"""
    
    def __init__(self, articles_path: str = "artifacts/articles.jsonl"):
        self.articles_path = Path(articles_path)
        self.article_cache = {}
        self.article_stats = {}
        self._load_articles()
        
    def _load_articles(self):
        """Load all articles into memory"""
        if not self.articles_path.exists():
            print(f"Warning: Articles file not found at {self.articles_path}")
            return
        
        with open(self.articles_path, 'r') as f:
            for line in f:
                if line.strip():
                    article = json.loads(line)
                    self.article_cache[article['uuid']] = article
        
        print(f"Loaded {len(self.article_cache)} articles")
    
    def extract_features(self, article_id: str, interaction_history: List[Dict] = None) -> Dict[str, Any]:
        """
        Extract features for an article
        
        Args:
            article_id: Article UUID
            interaction_history: Historical interactions with this article
            
        Returns:
            Dictionary of article features
        """
        if article_id not in self.article_cache:
            return self._get_default_features(article_id)
        
        article = self.article_cache[article_id]
        features = {}
        
        # Content features
        features.update(self._extract_content_features(article))
        
        # Statistical features from interactions
        if interaction_history:
            features.update(self._extract_statistical_features(article_id, interaction_history))
        else:
            features.update({
                'historical_ctr': 0,
                'historical_engagement_rate': 0,
                'historical_avg_dwell': 0,
                'impression_count': 0,
                'click_count': 0
            })
        
        return features
    
    def _extract_content_features(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content-based features"""
        text = article.get('text', '')
        topics = article.get('topics', [])
        
        # Text statistics
        words = text.split()
        sentences = text.split('.')
        
        return {
            'topics': topics,
            'num_topics': len(topics),
            'text_length': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'avg_sentence_length': np.mean([len(s.split()) for s in sentences if s.strip()]) if sentences else 0,
            'has_science_tech': 'science and technology' in topics,
            'has_health': 'health' in topics,
            'has_lifestyle': 'lifestyle and leisure' in topics,
            'has_education': 'education' in topics,
            'has_sport': 'sport' in topics,
            'has_environment': 'environment' in topics,
        }
    
    def _extract_statistical_features(self, article_id: str, interaction_history: List[Dict]) -> Dict[str, Any]:
        """Extract statistical features from interaction history"""
        clicks = 0
        impressions = 0
        dwell_times = []
        engagements = 0
        
        for interaction in interaction_history:
            for action in interaction.get('actions', []):
                if action['article_id'] == article_id:
                    impressions += 1
                    if action['clicked']:
                        clicks += 1
                        dwell_times.append(action['dwell_time_secs'])
                    if action['liked'] or action['shared'] or action['bookmarked']:
                        engagements += 1
        
        return {
            'historical_ctr': clicks / impressions if impressions > 0 else 0,
            'historical_engagement_rate': engagements / impressions if impressions > 0 else 0,
            'historical_avg_dwell': np.mean(dwell_times) if dwell_times else 0,
            'impression_count': impressions,
            'click_count': clicks,
            'engagement_count': engagements
        }
    
    def _get_default_features(self, article_id: str) -> Dict[str, Any]:
        """Return default features for unknown articles"""
        return {
            'article_id': article_id,
            'topics': [],
            'num_topics': 0,
            'text_length': 0,
            'word_count': 0,
            'is_unknown': True
        }
    
    def get_article_vector(self, article_id: str) -> np.ndarray:
        """
        Get numerical feature vector for article
        
        Returns fixed-size vector suitable for neural networks
        """
        features = self.extract_features(article_id)
        
        # One-hot encoding for topics (using common topics)
        topic_vector = np.array([
            features.get('has_science_tech', 0),
            features.get('has_health', 0),
            features.get('has_lifestyle', 0),
            features.get('has_education', 0),
            features.get('has_sport', 0),
            features.get('has_environment', 0),
        ], dtype=np.float32)
        
        # Statistical features
        stat_vector = np.array([
            features.get('word_count', 0) / 1000.0,  # Normalize
            features.get('historical_ctr', 0),
            features.get('historical_engagement_rate', 0),
            features.get('historical_avg_dwell', 0) / 60.0,  # Normalize to minutes
            features.get('impression_count', 0) / 100.0,  # Normalize
        ], dtype=np.float32)
        
        return np.concatenate([topic_vector, stat_vector])
    
    def get_article_metadata(self, article_id: str) -> Dict[str, Any]:
        """Get article metadata for logging"""
        if article_id in self.article_cache:
            article = self.article_cache[article_id]
            return {
                'uuid': article['uuid'],
                'topics': article.get('topics', []),
                'text': article.get('text', '')
            }
        return {'uuid': article_id, 'topics': [], 'text': ''}
    
    def get_articles_by_topic(self, topic: str, limit: int = 100) -> List[str]:
        """Get article IDs filtered by topic"""
        matching_articles = []
        for article_id, article in self.article_cache.items():
            if topic in article.get('topics', []):
                matching_articles.append(article_id)
                if len(matching_articles) >= limit:
                    break
        return matching_articles
    
    def compute_article_similarity(self, article_id1: str, article_id2: str) -> float:
        """Compute topic-based similarity between two articles"""
        if article_id1 not in self.article_cache or article_id2 not in self.article_cache:
            return 0.0
        
        topics1 = set(self.article_cache[article_id1].get('topics', []))
        topics2 = set(self.article_cache[article_id2].get('topics', []))
        
        if not topics1 or not topics2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(topics1 & topics2)
        union = len(topics1 | topics2)
        
        return intersection / union if union > 0 else 0.0
