"""
Time series forecasting models module.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')


class ARIMAModel:
    """ARIMA model wrapper for time series forecasting."""
    
    def __init__(self, order=(1, 1, 1)):
        """
        Initialize ARIMA model.
        
        Parameters:
        -----------
        order : tuple, default=(1, 1, 1)
            (p, d, q) order of the ARIMA model
        """
        self.order = order
        self.model = None
        self.fitted_model = None
    
    def fit(self, data):
        """
        Fit ARIMA model to the data.
        
        Parameters:
        -----------
        data : pd.Series
            Time series data
        """
        self.model = ARIMA(data, order=self.order)
        self.fitted_model = self.model.fit()
        return self
    
    def predict(self, steps=1):
        """
        Make predictions.
        
        Parameters:
        -----------
        steps : int, default=1
            Number of steps to forecast
        
        Returns:
        --------
        pd.Series
            Forecasted values
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast
    
    def get_summary(self):
        """Get model summary."""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first")
        return self.fitted_model.summary()


class SARIMAXModel:
    """SARIMAX model wrapper for time series forecasting."""
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0)):
        """
        Initialize SARIMAX model.
        
        Parameters:
        -----------
        order : tuple, default=(1, 1, 1)
            (p, d, q) order of the ARIMA model
        seasonal_order : tuple, default=(0, 0, 0, 0)
            (P, D, Q, s) seasonal order
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
    
    def fit(self, data):
        """
        Fit SARIMAX model to the data.
        
        Parameters:
        -----------
        data : pd.Series
            Time series data
        """
        self.model = SARIMAX(data, order=self.order, seasonal_order=self.seasonal_order)
        self.fitted_model = self.model.fit(disp=False)
        return self
    
    def predict(self, steps=1):
        """
        Make predictions.
        
        Parameters:
        -----------
        steps : int, default=1
            Number of steps to forecast
        
        Returns:
        --------
        pd.Series
            Forecasted values
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast


class ExponentialSmoothingModel:
    """Exponential Smoothing model wrapper."""
    
    def __init__(self, trend='add', seasonal='add', seasonal_periods=12):
        """
        Initialize Exponential Smoothing model.
        
        Parameters:
        -----------
        trend : str, default='add'
            Type of trend component ('add', 'mul', None)
        seasonal : str, default='add'
            Type of seasonal component ('add', 'mul', None)
        seasonal_periods : int, default=12
            Number of periods in a complete seasonal cycle
        """
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.fitted_model = None
    
    def fit(self, data):
        """
        Fit Exponential Smoothing model to the data.
        
        Parameters:
        -----------
        data : pd.Series
            Time series data
        """
        self.model = ExponentialSmoothing(
            data, 
            trend=self.trend, 
            seasonal=self.seasonal, 
            seasonal_periods=self.seasonal_periods
        )
        self.fitted_model = self.model.fit()
        return self
    
    def predict(self, steps=1):
        """
        Make predictions.
        
        Parameters:
        -----------
        steps : int, default=1
            Number of steps to forecast
        
        Returns:
        --------
        pd.Series
            Forecasted values
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast


class MovingAverageModel:
    """Simple Moving Average model."""
    
    def __init__(self, window=7):
        """
        Initialize Moving Average model.
        
        Parameters:
        -----------
        window : int, default=7
            Size of the moving window
        """
        self.window = window
        self.data = None
    
    def fit(self, data):
        """
        Fit Moving Average model to the data.
        
        Parameters:
        -----------
        data : pd.Series
            Time series data
        """
        self.data = data
        return self
    
    def predict(self, steps=1):
        """
        Make predictions.
        
        Parameters:
        -----------
        steps : int, default=1
            Number of steps to forecast
        
        Returns:
        --------
        np.array
            Forecasted values (simple prediction using last window mean)
        """
        if self.data is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Simple approach: use mean of last window for all future steps
        last_values = self.data.tail(self.window)
        prediction = last_values.mean()
        
        return np.array([prediction] * steps)


class LinearRegressionModel:
    """Linear Regression model for time series forecasting."""
    
    def __init__(self):
        """Initialize Linear Regression model."""
        self.model = LinearRegression()
        self.last_index = None
    
    def fit(self, data):
        """
        Fit Linear Regression model to the data.
        
        Parameters:
        -----------
        data : pd.Series
            Time series data
        """
        X = np.arange(len(data)).reshape(-1, 1)
        y = data.values
        self.model.fit(X, y)
        self.last_index = len(data)
        return self
    
    def predict(self, steps=1):
        """
        Make predictions.
        
        Parameters:
        -----------
        steps : int, default=1
            Number of steps to forecast
        
        Returns:
        --------
        np.array
            Forecasted values
        """
        if self.last_index is None:
            raise ValueError("Model must be fitted before prediction")
        
        future_X = np.arange(self.last_index, self.last_index + steps).reshape(-1, 1)
        predictions = self.model.predict(future_X)
        
        return predictions
