
# Recommendation System

This project implements a basic recommendation system using collaborative filtering techniques. It recommends products, movies, or articles based on user preferences.

## Project Structure
- `data/`: Contains the user-item interaction dataset.
- `src/`: Source code for data loading, collaborative filtering, and recommendation system.
- `notebooks/`: Jupyter notebook for exploratory data analysis.
- `main.py`: Entry point to run the recommendation system.

## Dependencies
- Python 3.x
- Scikit-learn
- Pandas
- NumPy

## How to Run
1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
2. Run the recommendation system:
    ```
    python main.py
    ```

## Dataset
The dataset contains user-item interactions with ratings.

| user_id | item_id | rating |
|---------|---------|--------|
| 1       | 10      | 4.5    |
| 1       | 20      | 5.0    |
| 2       | 10      | 3.0    |
| 3       | 30      | 2.0    |

