import numpy as np
from data_loader import load_data
from collaborative_filter import CollaborativeFiltering

class RecommenderSystem:
    def __init__(self, data_path):
        self.train_data, self.test_data = load_data(data_path)
        
    def train(self):
        """
        Train the recommender system using collaborative filtering.
        """
        # Create user-item interaction matrix
        user_item_matrix = self.train_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0).values
        
        # Initialize and train collaborative filtering model
        self.cf_model = CollaborativeFiltering()
        self.cf_model.fit(user_item_matrix)
    
    def get_recommendations(self, user_id):
        """
        Get recommendations for a specific user.
        """
        # Create the same user-item matrix for prediction
        user_item_matrix = self.train_data.pivot(index='user_id', columns='item_id', values='rating').fillna(0).values
        user_index = np.where(self.train_data['user_id'].unique() == user_id)[0][0]
        recommendations = self.cf_model.recommend(user_index, user_item_matrix)
        
        return recommendations

if __name__ == "__main__":
    recommender = RecommenderSystem(data_path='data/user_item_data.csv')
    recommender.train()
    user_id = 1  # Example user_id for recommendations
    print(f"Recommendations for user {user_id}: {recommender.get_recommendations(user_id)}")
