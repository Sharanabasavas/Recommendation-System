from src.recommender import RecommenderSystem

def main():
    recommender = RecommenderSystem(data_path='data/user_item_data.csv')
    recommender.train()
    
    user_id = int(input("Enter user ID for recommendations: "))
    recommendations = recommender.get_recommendations(user_id)
    print(f"Recommendations for user {user_id}: {recommendations}")

if __name__ == "__main__":
    main()
