import logging
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine, euclidean

logger = logging.getLogger(__name__)


class CrossViewMatcher:
    """
    It matches player identities across two different camera views using
    a feature-based cost matrix and the Hungarian algorithm.
    """
    def __init__(self, feature_weights: dict, max_cost_threshold: float = 0.8):
        """
        Initializes the matcher with feature weights.

        Args:
            feature_weights (dict): A dictionary weighting the importance of each feature.
                                    e.g., {'appearance': 0.5, 'field_coords': 0.5}
            max_cost_threshold (float): The maximum allowable cost for a match to be considered valid.
        """
        if not np.isclose(sum(feature_weights.values()), 1.0):
            logger.error("Feature weights must sum to 1.")
            raise ValueError("Feature weights must sum to 1.")
            
        self.weights = feature_weights
        self.max_cost = max_cost_threshold
        logger.info(f"CrossViewMatcher initialized with weights: {self.weights}")

    def calculate_cost_matrix(self, players1: list, players2: list) -> np.ndarray:
        """
        Calculates the cost matrix between two sets of players.

        Args:
            players_view_1 (list): List of player data from view 1.
            players_view_2 (list): List of player data from view 2.

        Returns:
            np.ndarray: An M x N matrix of costs, where M=len(players1) and N=len(players2).
        """
        num_players1 = len(players1)
        num_players2 = len(players2)
        cost_matrix = np.zeros((num_players1, num_players2))

        for i in range(num_players1):
            for j in range(num_players2):
                p1_features = players1[i]['features']
                p2_features = players2[j]['features']
                
                total_cost = 0
                
                if 'appearance' in self.weights:
                    app_cost = cosine(p1_features['appearance'], p2_features['appearance'])
                    total_cost += self.weights['appearance'] * app_cost
                
                if 'field_coords' in self.weights:
                    max_field_dist = 100
                    coord_cost = euclidean(p1_features['field_coords'], p2_features['field_coords']) / max_field_dist
                    total_cost += self.weights['field_coords'] * coord_cost
                
                if 'color_hist' in self.weights:
                    color_cost = chi_squared_distance(p1_features['color_hist'], p2_features['color_hist'])
                    total_cost += self.weights['color_hist'] * color_cost
                    
                cost_matrix[i, j] = total_cost
        
        return cost_matrix

    def match_players_in_frame(self, players_view_1: list, players_view_2: list) -> tuple:
        """
        Performs the matching for a single frame.

        Returns:
            A tuple containing:
            - matched_pairs (list): List of (player1_data, player2_data) tuples.
            - unmatched1 (list): List of unmatched players from view 1.
            - unmatched2 (list): List of unmatched players from view 2.
        """
        if not players_view_1 or not players_view_2:
            logger.warning("One list is empty (either player_view1 or player_view2)")
            return [], players_view_1, players_view_2
        
        cost_matrix = self.calculate_cost_matrix(players_view_1, players_view_2)
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matched_pairs = []
        unmatched1_indices = set(range(len(players_view_1)))
        unmatched2_indices = set(range(len(players_view_2)))
        
        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            if cost < self.max_cost:
                matched_pairs.append((players_view_1[r], players_view_2[c]))
                unmatched1_indices.discard(r)
                unmatched2_indices.discard(c)
                logger.debug(f"Matched Player_ID {players_view_1[r]['track_id']} in View 1 with Player_ID {players_view_2[c]['track_id']} in View 2 with cost {cost:.2f}")

        unmatched1 = [players_view_1[i] for i in unmatched1_indices]
        unmatched2 = [players_view_2[i] for i in unmatched2_indices]

        return matched_pairs, unmatched1, unmatched2
    

def chi_squared_distance(hist_a, hist_b, eps=1e-10):
    """Computes the Chi-Squared distance between two histograms."""
    return 0.5 * np.sum([((a - b) ** 2) / (a + b + eps) for (a, b) in zip(hist_a, hist_b)])