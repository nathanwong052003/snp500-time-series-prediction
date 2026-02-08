"""
Evaluation metrics module for time series forecasting.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_mse(y_true, y_pred):
    """
    Calculate Mean Squared Error.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    float
        MSE value
    """
    return mean_squared_error(y_true, y_pred)


def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    float
        RMSE value
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    float
        MAE value
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    float
        MAPE value (in percentage)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_r2(y_true, y_pred):
    """
    Calculate R-squared (coefficient of determination).
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    float
        R² value
    """
    return r2_score(y_true, y_pred)


def evaluate_forecast(y_true, y_pred, metrics=['mse', 'rmse', 'mae', 'mape', 'r2']):
    """
    Evaluate forecast using multiple metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    metrics : list, default=['mse', 'rmse', 'mae', 'mape', 'r2']
        List of metrics to calculate
    
    Returns:
    --------
    dict
        Dictionary with metric names and values
    """
    results = {}
    
    if 'mse' in metrics:
        results['MSE'] = calculate_mse(y_true, y_pred)
    
    if 'rmse' in metrics:
        results['RMSE'] = calculate_rmse(y_true, y_pred)
    
    if 'mae' in metrics:
        results['MAE'] = calculate_mae(y_true, y_pred)
    
    if 'mape' in metrics:
        results['MAPE'] = calculate_mape(y_true, y_pred)
    
    if 'r2' in metrics:
        results['R²'] = calculate_r2(y_true, y_pred)
    
    return results


def print_evaluation_results(results):
    """
    Print evaluation results in a formatted way.
    
    Parameters:
    -----------
    results : dict
        Dictionary with metric names and values
    """
    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for metric, value in results.items():
        print(f"{metric:10s}: {value:.4f}")
    print("=" * 50)


def compare_models(model_results):
    """
    Compare multiple models based on their evaluation results.
    
    Parameters:
    -----------
    model_results : dict
        Dictionary with model names as keys and evaluation results as values
    
    Returns:
    --------
    pd.DataFrame
        DataFrame comparing all models
    """
    df = pd.DataFrame(model_results).T
    return df
