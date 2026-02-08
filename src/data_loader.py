"""
Data loader module for fetching S&P 500 stock price data.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def fetch_sp500_data(start_date=None, end_date=None, period="5y"):
    """
    Fetch S&P 500 historical data using yfinance.
    
    Parameters:
    -----------
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
    period : str, default='5y'
        Period to download (e.g., '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with S&P 500 historical data
    """
    sp500 = yf.Ticker("^GSPC")
    
    if start_date and end_date:
        data = sp500.history(start=start_date, end=end_date)
    else:
        data = sp500.history(period=period)
    
    return data


def load_data(filepath):
    """
    Load S&P 500 data from a CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with the loaded data
    """
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return data


def save_data(data, filepath):
    """
    Save S&P 500 data to a CSV file.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame to save
    filepath : str
        Path where to save the CSV file
    """
    data.to_csv(filepath)
    print(f"Data saved to {filepath}")


def get_close_prices(data):
    """
    Extract close prices from the dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with stock data
    
    Returns:
    --------
    pd.Series
        Series with close prices
    """
    return data['Close']
