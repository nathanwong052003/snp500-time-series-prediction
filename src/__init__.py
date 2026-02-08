"""
S&P 500 Time Series Prediction Package

This package provides utilities for time series forecasting on S&P 500 stock prices.
"""

__version__ = '1.0.0'

from .data_loader import fetch_sp500_data, load_data, save_data, get_close_prices
from .preprocessing import (
    check_missing_values, 
    fill_missing_values,
    train_test_split_timeseries,
    normalize_data,
    create_lag_features,
    create_rolling_features
)
from .models import (
    ARIMAModel,
    SARIMAXModel,
    ExponentialSmoothingModel,
    MovingAverageModel,
    LinearRegressionModel
)
from .evaluation import (
    evaluate_forecast,
    print_evaluation_results,
    compare_models
)

__all__ = [
    # Data loading
    'fetch_sp500_data',
    'load_data',
    'save_data',
    'get_close_prices',
    
    # Preprocessing
    'check_missing_values',
    'fill_missing_values',
    'train_test_split_timeseries',
    'normalize_data',
    'create_lag_features',
    'create_rolling_features',
    
    # Models
    'ARIMAModel',
    'SARIMAXModel',
    'ExponentialSmoothingModel',
    'MovingAverageModel',
    'LinearRegressionModel',
    
    # Evaluation
    'evaluate_forecast',
    'print_evaluation_results',
    'compare_models',
]
