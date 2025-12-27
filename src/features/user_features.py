"""
User feature extraction for personalization
"""
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict, Counter
from datetime import datetime, timedelta


class UserFeatureExtractor:
    """Extract user-level features from interaction history"""
    
    def __init__(self):
        self.user_profiles = {}
        
    def extract_features(self, user_id: str, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract comprehensive user features from interaction history
        
        Returns:
            Dictionary containing user features including:
            - Topic preferences
            - Behavioral patterns
            - Engagement metrics
            - Temporal patterns
        """
        if not interactions:
            return self._get_default_features()
        
        features = {}
        
        # Topic-based features
        features.update(self._extract_topic_features(interactions))
        
        # Behavioral features
        features.update(self._extract_behavioral_features(interactions))
        
        # Engagement features
        features.update(self._extract_engagement_features(interactions))
        
        # Temporal features
        features.update(self._extract_temporal_features(interactions))
        
        # Contextual features
        features.update(self._extract_contextual_features(interactions))
        
        # Cache the profile
        self.user_profiles[user_id] = features
        
        return features
    
    def _extract_topic_features(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract topic preference features"""
        topic_clicks = defaultdict(int)
        topic_impressions = defaultdict(int)
        topic_dwell_times = defaultdict(list)
        topic_engagements = defaultdict(int)
        
        for interaction in interactions:
            for action in interaction['actions']:
                topics = action.get('topics', [])
                for topic in topics:
                    topic_impressions[topic] += 1
                    if action['clicked']:
                        topic_clicks[topic] += 1
                        topic_dwell_times[topic].append(action['dwell_time_secs'])
                    if action['liked'] or action['shared'] or action['bookmarked']:
                        topic_engagements[topic] += 1
        
        # Calculate topic affinities
        topic_affinities = {}
        for topic in topic_impressions:
            ctr = topic_clicks[topic] / topic_impressions[topic] if topic_impressions[topic] > 0 else 0
            avg_dwell = np.mean(topic_dwell_times[topic]) if topic_dwell_times[topic] else 0
            engagement_rate = topic_engagements[topic] / topic_impressions[topic] if topic_impressions[topic] > 0 else 0
            
            # Combined affinity score
            topic_affinities[topic] = {
                'ctr': ctr,
                'avg_dwell_time': avg_dwell,
                'engagement_rate': engagement_rate,
                'impressions': topic_impressions[topic],
                'affinity_score': (ctr * 0.3 + (avg_dwell / 60) * 0.3 + engagement_rate * 0.4)
            }
        
        # Get top topics
        top_topics = sorted(topic_affinities.items(), 
                           key=lambda x: x[1]['affinity_score'], 
                           reverse=True)[:5]
        
        return {
            'topic_affinities': topic_affinities,
            'top_topics': [t[0] for t in top_topics],
            'topic_diversity': len(topic_affinities),
            'dominant_topic': top_topics[0][0] if top_topics else None,
            'dominant_topic_score': top_topics[0][1]['affinity_score'] if top_topics else 0
        }
    
    def _extract_behavioral_features(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract behavioral patterns"""
        # Compute clicks from actions (handle both old and new format)
        total_clicks = 0
        queries_with_clicks = 0
        total_queries = len(interactions)
        
        for i in interactions:
            if 'num_clicks' in i:
                total_clicks += i['num_clicks']
            else:
                interaction_clicks = sum(1 for a in i.get('actions', []) if a.get('clicked', False))
                total_clicks += interaction_clicks
                if interaction_clicks > 0:
                    queries_with_clicks += 1
        
        click_positions = []
        dwell_times = []
        
        for interaction in interactions:
            for action in interaction.get('actions', []):
                if action.get('clicked', False):
                    click_positions.append(action.get('position', 0))
                    dwell = action.get('dwell_time_secs', 0)
                    if dwell > 0:
                        dwell_times.append(dwell)
        
        return {
            'overall_ctr': queries_with_clicks / total_queries if total_queries > 0 else 0,
            'avg_click_position': np.mean(click_positions) if click_positions else -1,
            'avg_dwell_time': np.mean(dwell_times) if dwell_times else 0,
            'std_dwell_time': np.std(dwell_times) if len(dwell_times) > 1 else 0,
            'total_queries': total_queries,
            'clicks_per_query': total_clicks / total_queries if total_queries > 0 else 0,
            'exploration_rate': len(set(click_positions)) / len(click_positions) if click_positions else 0
        }
    
    def _extract_engagement_features(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract engagement metrics"""
        likes = shares = bookmarks = 0
        total_engagements = 0
        total_clicks = 0
        
        for interaction in interactions:
            for action in interaction.get('actions', []):
                if action.get('liked', False):
                    likes += 1
                    total_engagements += 1
                if action.get('shared', False):
                    shares += 1
                    total_engagements += 1
                if action.get('bookmarked', False):
                    bookmarks += 1
                    total_engagements += 1
                if action.get('clicked', False):
                    total_clicks += 1
            # Also check for num_engagements/num_clicks in old format
            total_engagements += interaction.get('num_engagements', 0)
            total_clicks += interaction.get('num_clicks', 0) if 'num_clicks' in interaction else 0
        
        return {
            'like_rate': likes / max(total_clicks, 1),
            'share_rate': shares / max(total_clicks, 1),
            'bookmark_rate': bookmarks / max(total_clicks, 1),
            'engagement_rate': total_engagements / max(total_clicks, 1),
            'total_likes': likes,
            'total_shares': shares,
            'total_bookmarks': bookmarks
        }
    
    def _extract_temporal_features(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract temporal patterns"""
        timestamps = []
        for i in interactions:
            if 'timestamp' in i:
                try:
                    timestamps.append(datetime.fromisoformat(i['timestamp']))
                except:
                    pass
        
        if len(timestamps) < 2:
            return {
                'avg_session_gap_hours': 0,
                'activity_hours': [],
                'is_active_user': True,  # Assume active if we have data
                'recency_days': 0,
                'peak_activity_hour': -1,
                'activity_span_days': 0
            }
        
        # Session gaps
        timestamps_sorted = sorted(timestamps)
        gaps = [(timestamps_sorted[i+1] - timestamps_sorted[i]).total_seconds() / 3600 
                for i in range(len(timestamps_sorted) - 1)]
        
        # Activity hours
        hours = [t.hour for t in timestamps]
        hour_counter = Counter(hours)
        
        # Recency
        most_recent = max(timestamps)
        try:
            recency = (datetime.utcnow() - most_recent).days
        except:
            recency = 0
        
        return {
            'avg_session_gap_hours': np.mean(gaps) if gaps else 0,
            'activity_hours': list(hour_counter.keys()),
            'peak_activity_hour': hour_counter.most_common(1)[0][0] if hour_counter else -1,
            'is_active_user': recency < 7,
            'recency_days': recency,
            'activity_span_days': (max(timestamps) - min(timestamps)).days
        }
    
    def _extract_contextual_features(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract contextual features like query patterns"""
        queries = [i['query_text'] for i in interactions]
        query_lengths = [len(q.split()) for q in queries]
        
        return {
            'avg_query_length': np.mean(query_lengths) if query_lengths else 0,
            'query_diversity': len(set(queries)) / len(queries) if queries else 0,
            'repeat_query_rate': 1 - (len(set(queries)) / len(queries)) if queries else 0
        }
    
    def _get_default_features(self) -> Dict[str, Any]:
        """Return default features for cold-start users"""
        return {
            'topic_affinities': {},
            'top_topics': [],
            'topic_diversity': 0,
            'dominant_topic': None,
            'dominant_topic_score': 0,
            'overall_ctr': 0,
            'avg_click_position': -1,
            'avg_dwell_time': 0,
            'std_dwell_time': 0,
            'total_queries': 0,
            'clicks_per_query': 0,
            'exploration_rate': 0,
            'like_rate': 0,
            'share_rate': 0,
            'bookmark_rate': 0,
            'engagement_rate': 0,
            'is_cold_start': True
        }
    
    def get_user_vector(self, user_id: str) -> np.ndarray:
        """
        Get numerical feature vector for user (for ML models)
        
        Returns fixed-size vector suitable for neural networks
        """
        if user_id not in self.user_profiles:
            features = self._get_default_features()
        else:
            features = self.user_profiles[user_id]
        
        # Create fixed-size vector (can be expanded)
        vector = np.array([
            features.get('overall_ctr', 0),
            features.get('avg_click_position', 0),
            features.get('avg_dwell_time', 0),
            features.get('exploration_rate', 0),
            features.get('engagement_rate', 0),
            features.get('like_rate', 0),
            features.get('share_rate', 0),
            features.get('bookmark_rate', 0),
            features.get('query_diversity', 0),
            features.get('recency_days', 0) / 30.0,  # Normalize
            features.get('is_active_user', False),
        ], dtype=np.float32)
        
        return vector
