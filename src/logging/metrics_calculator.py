"""
Metrics calculator for ranking evaluation
"""
import numpy as np
from typing import List, Dict, Any
from collections import defaultdict


class MetricsCalculator:
    """Calculate engagement and ranking metrics"""
    
    @staticmethod
    def calculate_ctr(interactions: List[Dict[str, Any]]) -> float:
        """
        Calculate Click-Through Rate (impression-level).
        CTR = total clicks / total impressions
        This is the standard definition used in industry.
        """
        if not interactions:
            return 0.0
        
        total_impressions = 0
        total_clicks = 0
        
        for interaction in interactions:
            for action in interaction.get('actions', []):
                total_impressions += 1
                if action.get('clicked', False):
                    total_clicks += 1
        
        return total_clicks / total_impressions if total_impressions > 0 else 0.0
    
    @staticmethod
    def calculate_query_ctr(interactions: List[Dict[str, Any]]) -> float:
        """
        Calculate Query-level CTR (% of queries with at least one click).
        Useful when not all results are viewed by users.
        """
        if not interactions:
            return 0.0
        
        queries_with_clicks = sum(
            1 for i in interactions
            if any(a.get('clicked', False) for a in i.get('actions', []))
        )
        
        return queries_with_clicks / len(interactions)
    
    @staticmethod
    def calculate_position_ctr(interactions: List[Dict[str, Any]]) -> Dict[int, float]:
        """Calculate CTR by position"""
        position_clicks = defaultdict(int)
        position_impressions = defaultdict(int)
        
        for interaction in interactions:
            for action in interaction['actions']:
                pos = action['position']
                position_impressions[pos] += 1
                if action['clicked']:
                    position_clicks[pos] += 1
        
        position_ctr = {}
        for pos in position_impressions:
            position_ctr[pos] = position_clicks[pos] / position_impressions[pos]
        
        return dict(sorted(position_ctr.items()))
    
    @staticmethod
    def calculate_mrr(interactions: List[Dict[str, Any]]) -> float:
        """Calculate Mean Reciprocal Rank of first click"""
        reciprocal_ranks = []
        
        for interaction in interactions:
            for action in interaction['actions']:
                if action['clicked']:
                    reciprocal_ranks.append(1.0 / (action['position'] + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    @staticmethod
    def calculate_ndcg(interactions: List[Dict[str, Any]], k: int = 10) -> float:
        """
        Calculate NDCG@k using implicit feedback
        Relevance: 0 (skip), 1 (click), 2 (click+dwell>10s), 3 (engagement)
        """
        ndcg_scores = []
        
        for interaction in interactions:
            relevances = []
            for action in interaction['actions'][:k]:
                if action.get('bookmarked') or action.get('shared') or action.get('liked'):
                    rel = 3
                elif action.get('clicked') and action.get('dwell_time_secs', 0) > 10:
                    rel = 2
                elif action.get('clicked'):
                    rel = 1
                else:
                    rel = 0
                relevances.append(rel)
            
            if not relevances or sum(relevances) == 0:
                ndcg_scores.append(0.0)
                continue
            
            # DCG
            dcg = relevances[0] + sum(
                rel / np.log2(idx + 2) for idx, rel in enumerate(relevances[1:], start=1)
            )
            
            # Ideal DCG
            ideal_relevances = sorted(relevances, reverse=True)
            idcg = ideal_relevances[0] + sum(
                rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevances[1:], start=1)
            )
            
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    @staticmethod
    def calculate_engagement_rate(interactions: List[Dict[str, Any]]) -> float:
        """Calculate overall engagement rate (likes, shares, bookmarks)"""
        if not interactions:
            return 0.0
        
        total_results = sum(len(i['actions']) for i in interactions)
        total_engagements = sum(
            sum(1 for a in i['actions'] if a.get('liked') or a.get('shared') or a.get('bookmarked'))
            for i in interactions
        )
        
        return total_engagements / total_results if total_results > 0 else 0.0
    
    @staticmethod
    def calculate_avg_dwell_time(interactions: List[Dict[str, Any]]) -> float:
        """Calculate average dwell time on clicked articles"""
        dwell_times = []
        
        for interaction in interactions:
            for action in interaction['actions']:
                if action.get('clicked') and action.get('dwell_time_secs', 0) > 0:
                    dwell_times.append(action['dwell_time_secs'])
        
        return np.mean(dwell_times) if dwell_times else 0.0
    
    @staticmethod
    def calculate_skip_rate(interactions: List[Dict[str, Any]]) -> float:
        """Calculate rate of skipped articles"""
        if not interactions:
            return 0.0
        
        total_results = sum(len(i['actions']) for i in interactions)
        total_skips = sum(
            sum(1 for a in i['actions'] if a.get('skipped', not a.get('clicked')))
            for i in interactions
        )
        
        return total_skips / total_results if total_results > 0 else 0.0
    
    @staticmethod
    def get_all_metrics(interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate all metrics at once"""
        return {
            'ctr': MetricsCalculator.calculate_ctr(interactions),  # Standard impression-level CTR
            'query_ctr': MetricsCalculator.calculate_query_ctr(interactions),  # Query-level CTR
            'mrr': MetricsCalculator.calculate_mrr(interactions),
            'ndcg@5': MetricsCalculator.calculate_ndcg(interactions, k=5),
            'ndcg@10': MetricsCalculator.calculate_ndcg(interactions, k=10),
            'engagement_rate': MetricsCalculator.calculate_engagement_rate(interactions),
            'avg_dwell_time': MetricsCalculator.calculate_avg_dwell_time(interactions),
            'skip_rate': MetricsCalculator.calculate_skip_rate(interactions),
            'position_ctr': MetricsCalculator.calculate_position_ctr(interactions)
        }
