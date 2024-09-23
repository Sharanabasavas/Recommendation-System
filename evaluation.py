from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_recommendations(y_true, y_pred):
    """
    Evaluate the recommendations using RMSE.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return rmse
