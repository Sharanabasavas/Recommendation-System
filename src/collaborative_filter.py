import numpy as np
from sklearn.neighbors import NearestNeighbors

class CollaborativeFiltering:
    def __init__(self, metric='cosine', n_neighbors=5):
        """
        Initialize collaborative filtering with the desired similarity metric.
        """
        self.model = NearestNeighbors(metric=metric, n_neighbors=n_neighbors)
        
    def fit(self, user_item_matrix):
        """
        Fit the model with user-item interaction matrix.
        """
        self.model.fit(user_item_matrix)
        
    def recommend(self, user_index, user_item_matrix, n_recommendations=5):
        """
        Recommend items for a user based on similar users.
        """
        distances, indices = self.model.kneighbors(user_item_matrix[user_index], n_neighbors=n_recommendations)
        return indices.flatten()
