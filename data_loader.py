import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Load and preprocess the user-item interaction data.
    """
    data = pd.read_csv(file_path)
    
    # You can split the data into train and test sets for validation
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    return train_data, test_data
