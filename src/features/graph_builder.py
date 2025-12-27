"""
Graph builder for User-Article interactions
Creates heterogeneous graphs for GNN-based personalization
"""
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Tuple
from collections import defaultdict


class UserArticleGraphBuilder:
    """
    Build heterogeneous graph for user-article interactions
    
    Graph structure:
    - Nodes: Users, Articles, Topics
    - Edges: 
        - User -> Article (click, like, share, bookmark with weights)
        - Article -> Topic (belongs_to)
        - User -> Topic (interested_in, derived from interactions)
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.user_nodes = set()
        self.article_nodes = set()
        self.topic_nodes = set()
        self.interaction_counts = {}
        
    def build_graph(self, interactions: List[Dict[str, Any]], 
                    article_metadata: Dict[str, Dict[str, Any]]) -> nx.DiGraph:
        """
        Build graph from interaction data
        
        Args:
            interactions: List of interaction logs
            article_metadata: Dictionary mapping article_id to metadata (including topics)
            
        Returns:
            NetworkX DiGraph with heterogeneous nodes and edges
        """
        self.graph.clear()
        self.user_nodes.clear()
        self.article_nodes.clear()
        self.topic_nodes.clear()
        
        # First pass: Add nodes
        for interaction in interactions:
            user_id = interaction['user_id']
            self._add_user_node(user_id)
            
            for action in interaction['actions']:
                article_id = action['article_id']
                self._add_article_node(article_id)
                
                # Add topics
                if article_id in article_metadata:
                    topics = article_metadata[article_id].get('topics', [])
                    for topic in topics:
                        self._add_topic_node(topic)
                        self._add_article_topic_edge(article_id, topic)
        
        # Second pass: Add edges from interactions
        for interaction in interactions:
            user_id = interaction['user_id']
            
            for action in interaction['actions']:
                article_id = action['article_id']
                
                # Calculate edge weight based on interaction type
                weight = self._calculate_interaction_weight(action)
                
                # Always add edges (even for low weights) to densify graph
                if weight > 0:
                    self._add_user_article_edge(user_id, article_id, action, weight)
                    
                    # Add user-topic edges (only for stronger signals)
                    if weight >= 0.1 and article_id in article_metadata:
                        topics = article_metadata[article_id].get('topics', [])
                        for topic in topics:
                            self._add_user_topic_edge(user_id, topic, weight)
        
        print(f"Graph built: {len(self.user_nodes)} users, {len(self.article_nodes)} articles, "
              f"{len(self.topic_nodes)} topics, {self.graph.number_of_edges()} edges")
        
        return self.graph
    
    def _add_user_node(self, user_id: str):
        """Add user node to graph"""
        if user_id not in self.user_nodes:
            self.graph.add_node(user_id, node_type='user')
            self.user_nodes.add(user_id)
    
    def _add_article_node(self, article_id: str):
        """Add article node to graph"""
        if article_id not in self.article_nodes:
            self.graph.add_node(article_id, node_type='article')
            self.article_nodes.add(article_id)
    
    def _add_topic_node(self, topic: str):
        """Add topic node to graph"""
        if topic not in self.topic_nodes:
            self.graph.add_node(topic, node_type='topic')
            self.topic_nodes.add(topic)
    
    def _calculate_interaction_weight(self, action: Dict[str, Any]) -> float:
        """
        Calculate interaction weight based on action type
        
        Weights:
        - Skip: 0.05 (still saw the article, slight signal)
        - Impression (not clicked): 0.1
        - Click: 0.3
        - Click + Dwell (>10s): 0.5
        - Like: 0.7
        - Share: 0.8
        - Bookmark: 0.9
        """
        # Base weight for just seeing the article (impression)
        weight = 0.05
        
        if action['skipped']:
            return weight  # Small negative-ish signal but still counts
        
        # Non-skipped impression gets slightly more weight
        weight = 0.1
        
        if action['clicked']:
            weight = 0.3
            if action['dwell_time_secs'] > 10:
                weight = 0.5
            if action['dwell_time_secs'] > 30:
                weight = 0.6
        
        if action['liked']:
            weight = max(weight, 0.7)
        
        if action['shared']:
            weight = max(weight, 0.8)
        
        if action['bookmarked']:
            weight = max(weight, 0.9)
        
        return weight
    
    def _add_user_article_edge(self, user_id: str, article_id: str, 
                               action: Dict[str, Any], weight: float):
        """Add or update user-article edge"""
        if self.graph.has_edge(user_id, article_id):
            # Update existing edge (accumulate weights)
            edge_data = self.graph[user_id][article_id]
            edge_data['weight'] += weight
            edge_data['interaction_count'] += 1
            edge_data['total_dwell_time'] += action['dwell_time_secs']
        else:
            # Add new edge
            self.graph.add_edge(
                user_id, article_id,
                edge_type='interacts_with',
                weight=weight,
                interaction_count=1,
                total_dwell_time=action['dwell_time_secs'],
                clicked=action['clicked'],
                liked=action['liked'],
                shared=action['shared'],
                bookmarked=action['bookmarked']
            )
    
    def _add_article_topic_edge(self, article_id: str, topic: str):
        """Add article-topic edge"""
        if not self.graph.has_edge(article_id, topic):
            self.graph.add_edge(article_id, topic, edge_type='belongs_to', weight=1.0)
    
    def _add_user_topic_edge(self, user_id: str, topic: str, weight: float):
        """Add or update user-topic edge (interest)"""
        if self.graph.has_edge(user_id, topic):
            edge_data = self.graph[user_id][topic]
            edge_data['weight'] += weight
            edge_data['interaction_count'] += 1
        else:
            self.graph.add_edge(
                user_id, topic,
                edge_type='interested_in',
                weight=weight,
                interaction_count=1
            )
    
    def get_user_neighborhood(self, user_id: str, hops: int = 2) -> nx.DiGraph:
        """Get subgraph around a user (k-hop neighborhood)"""
        if user_id not in self.graph:
            return nx.DiGraph()
        
        # Get nodes within k hops
        nodes = {user_id}
        current_nodes = {user_id}
        
        for _ in range(hops):
            next_nodes = set()
            for node in current_nodes:
                # Outgoing edges
                next_nodes.update(self.graph.successors(node))
                # Incoming edges
                next_nodes.update(self.graph.predecessors(node))
            nodes.update(next_nodes)
            current_nodes = next_nodes
        
        return self.graph.subgraph(nodes).copy()
    
    def get_article_neighborhood(self, article_id: str, hops: int = 2) -> nx.DiGraph:
        """Get subgraph around an article"""
        if article_id not in self.graph:
            return nx.DiGraph()
        
        nodes = {article_id}
        current_nodes = {article_id}
        
        for _ in range(hops):
            next_nodes = set()
            for node in current_nodes:
                next_nodes.update(self.graph.successors(node))
                next_nodes.update(self.graph.predecessors(node))
            nodes.update(next_nodes)
            current_nodes = next_nodes
        
        return self.graph.subgraph(nodes).copy()
    
    def get_similar_users(self, user_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find similar users based on topic interests"""
        if user_id not in self.graph:
            return []
        
        # Get user's interested topics
        user_topics = {}
        for neighbor in self.graph.successors(user_id):
            if self.graph.nodes[neighbor].get('node_type') == 'topic':
                edge_data = self.graph[user_id][neighbor]
                user_topics[neighbor] = edge_data['weight']
        
        if not user_topics:
            return []
        
        # Find other users interested in similar topics
        user_similarities = defaultdict(float)
        
        for topic in user_topics:
            # Find other users interested in this topic
            for other_user in self.graph.predecessors(topic):
                if (self.graph.nodes[other_user].get('node_type') == 'user' and 
                    other_user != user_id):
                    edge_data = self.graph[other_user][topic]
                    # Weighted by both users' interest
                    similarity = min(user_topics[topic], edge_data['weight'])
                    user_similarities[other_user] += similarity
        
        # Sort by similarity
        similar_users = sorted(user_similarities.items(), 
                              key=lambda x: x[1], 
                              reverse=True)[:top_k]
        
        return similar_users
    
    def get_similar_articles(self, article_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find similar articles based on topics and user interactions"""
        if article_id not in self.graph:
            return []
        
        # Get article's topics
        article_topics = set()
        for neighbor in self.graph.successors(article_id):
            if self.graph.nodes[neighbor].get('node_type') == 'topic':
                article_topics.add(neighbor)
        
        if not article_topics:
            return []
        
        # Find articles with similar topics
        article_scores = defaultdict(float)
        
        for topic in article_topics:
            for other_article in self.graph.predecessors(topic):
                if (self.graph.nodes[other_article].get('node_type') == 'article' and 
                    other_article != article_id):
                    article_scores[other_article] += 1.0  # Jaccard-like similarity
        
        # Normalize by union of topics
        for other_article in list(article_scores.keys()):
            other_topics = set()
            for neighbor in self.graph.successors(other_article):
                if self.graph.nodes[neighbor].get('node_type') == 'topic':
                    other_topics.add(neighbor)
            
            union_size = len(article_topics | other_topics)
            if union_size > 0:
                article_scores[other_article] /= union_size
        
        # Sort by similarity
        similar_articles = sorted(article_scores.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:top_k]
        
        return similar_articles
    
    def get_recommended_articles_for_user(self, user_id: str, 
                                         exclude_articles: set = None,
                                         top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get article recommendations using graph-based collaborative filtering
        
        Strategy:
        1. Find articles liked by similar users
        2. Find articles similar to user's liked articles
        3. Combine scores
        """
        if user_id not in self.graph:
            return []
        
        exclude_articles = exclude_articles or set()
        article_scores = defaultdict(float)
        
        # Strategy 1: Similar users' articles
        similar_users = self.get_similar_users(user_id, top_k=20)
        for similar_user, similarity in similar_users:
            for article in self.graph.successors(similar_user):
                if (self.graph.nodes[article].get('node_type') == 'article' and
                    article not in exclude_articles):
                    edge_data = self.graph[similar_user][article]
                    article_scores[article] += similarity * edge_data['weight']
        
        # Strategy 2: User's liked articles -> similar articles
        user_articles = []
        for article in self.graph.successors(user_id):
            if self.graph.nodes[article].get('node_type') == 'article':
                edge_data = self.graph[user_id][article]
                user_articles.append((article, edge_data['weight']))
        
        for user_article, user_weight in user_articles:
            similar_articles = self.get_similar_articles(user_article, top_k=20)
            for similar_article, similarity in similar_articles:
                if similar_article not in exclude_articles:
                    article_scores[similar_article] += user_weight * similarity
        
        # Sort by combined score
        recommendations = sorted(article_scores.items(), 
                                key=lambda x: x[1], 
                                reverse=True)[:top_k]
        
        return recommendations
    
    def export_for_gnn(self) -> Dict[str, Any]:
        """
        Export graph in format suitable for PyTorch Geometric
        
        Returns dictionary with:
        - node_features: Dict of node type to feature matrix
        - edge_index: Dict of edge type to edge indices
        - edge_features: Dict of edge type to edge features
        """
        # Create node mappings
        user_to_idx = {user: idx for idx, user in enumerate(sorted(self.user_nodes))}
        article_to_idx = {article: idx for idx, article in enumerate(sorted(self.article_nodes))}
        topic_to_idx = {topic: idx for idx, topic in enumerate(sorted(self.topic_nodes))}
        
        # Prepare edge indices by type
        user_article_edges = []
        article_topic_edges = []
        user_topic_edges = []
        
        user_article_weights = []
        article_topic_weights = []
        user_topic_weights = []
        
        for u, v, data in self.graph.edges(data=True):
            edge_type = data.get('edge_type')
            weight = data.get('weight', 1.0)
            
            if edge_type == 'interacts_with':
                user_article_edges.append([user_to_idx[u], article_to_idx[v]])
                user_article_weights.append(weight)
            elif edge_type == 'belongs_to':
                article_topic_edges.append([article_to_idx[u], topic_to_idx[v]])
                article_topic_weights.append(weight)
            elif edge_type == 'interested_in':
                user_topic_edges.append([user_to_idx[u], topic_to_idx[v]])
                user_topic_weights.append(weight)
        
        return {
            'num_users': len(self.user_nodes),
            'num_articles': len(self.article_nodes),
            'num_topics': len(self.topic_nodes),
            'user_to_idx': user_to_idx,
            'article_to_idx': article_to_idx,
            'topic_to_idx': topic_to_idx,
            'edge_index': {
                'user_article': np.array(user_article_edges).T if user_article_edges else np.array([[], []]),
                'article_topic': np.array(article_topic_edges).T if article_topic_edges else np.array([[], []]),
                'user_topic': np.array(user_topic_edges).T if user_topic_edges else np.array([[], []])
            },
            'edge_weights': {
                'user_article': np.array(user_article_weights) if user_article_weights else np.array([]),
                'article_topic': np.array(article_topic_weights) if article_topic_weights else np.array([]),
                'user_topic': np.array(user_topic_weights) if user_topic_weights else np.array([])
            }
        }
