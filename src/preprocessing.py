"""
Preprocessing module for time series data preparation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def check_missing_values(data):
    """
    Check for missing values in the dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Data to check
    
    Returns:
    --------
    pd.Series
        Count of missing values per column
    """
    return data.isnull().sum()


def fill_missing_values(data, method='ffill'):
    """
    Fill missing values in the dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Data with missing values
    method : str, default='ffill'
        Method to fill missing values ('ffill', 'bfill', 'interpolate')
    
    Returns:
    --------
    pd.DataFrame or pd.Series
        Data with filled missing values
    """
    if method == 'interpolate':
        return data.interpolate()
    else:
        return data.fillna(method=method)


def create_lag_features(data, lags=[1, 2, 3, 7, 14, 30]):
    """
    Create lag features for time series.
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    lags : list, default=[1, 2, 3, 7, 14, 30]
        List of lag periods
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with lag features
    """
    df = pd.DataFrame(data)
    col_name = data.name if data.name else 'value'
    
    for lag in lags:
        df[f'{col_name}_lag_{lag}'] = data.shift(lag)
    
    return df


def create_rolling_features(data, windows=[7, 14, 30]):
    """
    Create rolling window features.
    
    Parameters:
    -----------
    data : pd.Series
        Time series data
    windows : list, default=[7, 14, 30]
        List of window sizes
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with rolling features
    """
    df = pd.DataFrame(data)
    col_name = data.name if data.name else 'value'
    
    for window in windows:
        df[f'{col_name}_rolling_mean_{window}'] = data.rolling(window=window).mean()
        df[f'{col_name}_rolling_std_{window}'] = data.rolling(window=window).std()
    
    return df


def normalize_data(data, scaler=None):
    """
    Normalize data using MinMaxScaler.
    
    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Data to normalize
    scaler : MinMaxScaler, optional
        Fitted scaler. If None, a new scaler is created and fitted.
    
    Returns:
    --------
    tuple
        (normalized_data, scaler)
    """
    if scaler is None:
        scaler = MinMaxScaler()
        
    if isinstance(data, pd.Series):
        values = data.values.reshape(-1, 1)
        normalized_values = scaler.fit_transform(values)
        normalized_data = pd.Series(normalized_values.flatten(), index=data.index)
    else:
        normalized_values = scaler.fit_transform(data)
        normalized_data = pd.DataFrame(normalized_values, index=data.index, columns=data.columns)
    
    return normalized_data, scaler


def train_test_split_timeseries(data, test_size=0.2):
    """
    Split time series data into train and test sets.
    
    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Time series data
    test_size : float, default=0.2
        Proportion of data to use for testing
    
    Returns:
    --------
    tuple
        (train_data, test_data)
    """
    split_idx = int(len(data) * (1 - test_size))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    return train_data, test_data
