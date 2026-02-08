# Getting Started with S&P 500 Time Series Forecasting

This guide will help you get started with the S&P 500 time series forecasting project.

## Quick Start

### 1. Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Test Script

To verify everything is working correctly:

```bash
python test_implementation.py
```

This will test all the core functionality of the modules.

### 3. Run the Example Script

To see a complete example of using the forecasting tools:

```bash
python example_usage.py
```

This will:
- Load/create sample S&P 500 data
- Train multiple forecasting models
- Evaluate and compare model performance
- Generate predictions
- Create visualizations

### 4. Explore the Jupyter Notebook

For a comprehensive, interactive analysis:

```bash
jupyter notebook notebooks/sp500_forecasting.ipynb
```

Or if you prefer JupyterLab:

```bash
jupyter lab
```

Then navigate to `notebooks/sp500_forecasting.ipynb` and run the cells.

## Project Structure

```
snp500-time-series-prediction/
├── data/                       # Data storage directory
│   └── .gitkeep               # Keeps directory in git
├── notebooks/                  # Jupyter notebooks
│   └── sp500_forecasting.ipynb # Main analysis notebook
├── src/                        # Source code modules
│   ├── __init__.py            # Package initialization
│   ├── data_loader.py         # Data fetching utilities
│   ├── preprocessing.py       # Data preprocessing functions
│   ├── models.py              # Time series models
│   └── evaluation.py          # Evaluation metrics
├── .gitignore                 # Git ignore file
├── README.md                  # Main documentation
├── GETTING_STARTED.md         # This file
├── requirements.txt           # Python dependencies
├── test_implementation.py     # Test script
└── example_usage.py           # Example usage script
```

## Available Models

1. **Moving Average** - Simple baseline model
2. **Linear Regression** - Trend-based forecasting
3. **ARIMA** - AutoRegressive Integrated Moving Average
4. **SARIMAX** - Seasonal ARIMA with eXogenous factors
5. **Exponential Smoothing** - Weighted average model

## Basic Usage Example

```python
from src.data_loader import fetch_sp500_data
from src.preprocessing import train_test_split_timeseries
from src.models import ARIMAModel
from src.evaluation import evaluate_forecast

# Fetch data
data = fetch_sp500_data(period="2y")

# Prepare data
close_prices = data['Close']
train, test = train_test_split_timeseries(close_prices, test_size=0.2)

# Train model
model = ARIMAModel(order=(5, 1, 0))
model.fit(train)

# Make predictions
predictions = model.predict(steps=len(test))

# Evaluate
results = evaluate_forecast(test.values, predictions)
print(results)
```

## Working with Your Own Data

To use your own stock data:

```python
from src.data_loader import fetch_sp500_data, save_data

# Fetch data for a specific period
data = fetch_sp500_data(
    start_date="2020-01-01",
    end_date="2024-12-31"
)

# Save for later use
save_data(data, "data/my_sp500_data.csv")

# Load previously saved data
from src.data_loader import load_data
loaded_data = load_data("data/my_sp500_data.csv")
```

## Customizing Models

### ARIMA Model

```python
from src.models import ARIMAModel

# Customize ARIMA order (p, d, q)
model = ARIMAModel(order=(5, 1, 2))
model.fit(train_data)
```

### Moving Average

```python
from src.models import MovingAverageModel

# Customize window size
model = MovingAverageModel(window=14)
model.fit(train_data)
```

### Exponential Smoothing

```python
from src.models import ExponentialSmoothingModel

# Customize trend and seasonal components
model = ExponentialSmoothingModel(
    trend='add',
    seasonal='mul',
    seasonal_periods=12
)
model.fit(train_data)
```

## Evaluation Metrics

The evaluation module provides several metrics:

- **MSE** - Mean Squared Error
- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **MAPE** - Mean Absolute Percentage Error
- **R²** - Coefficient of Determination

```python
from src.evaluation import evaluate_forecast, print_evaluation_results

results = evaluate_forecast(
    y_true=test_data.values,
    y_pred=predictions,
    metrics=['mse', 'rmse', 'mae', 'mape', 'r2']
)

print_evaluation_results(results)
```

## Comparing Multiple Models

```python
from src.evaluation import compare_models

# Train multiple models and get their results
model_results = {
    'ARIMA': arima_results,
    'Linear Regression': lr_results,
    'Moving Average': ma_results
}

# Compare
comparison_df = compare_models(model_results)
print(comparison_df)

# Find best model
best_model = comparison_df['RMSE'].idxmin()
print(f"Best model: {best_model}")
```

## Tips and Best Practices

1. **Start with simple models** - Moving Average and Linear Regression are good baselines
2. **Tune hyperparameters** - ARIMA order (p,d,q) can significantly impact performance
3. **Use appropriate test set size** - Typically 20-30% for time series
4. **Consider seasonality** - Use SARIMAX if your data has seasonal patterns
5. **Check for stationarity** - ARIMA works best with stationary data
6. **Validate assumptions** - Check residuals for autocorrelation

## Troubleshooting

### Import Errors

If you get import errors:
```bash
# Make sure you're in the project root directory
cd /path/to/snp500-time-series-prediction

# Install dependencies
pip install -r requirements.txt
```

### Data Fetching Issues

If you can't fetch data from Yahoo Finance:
- Check your internet connection
- Yahoo Finance API might be temporarily unavailable
- Use previously saved data or create mock data for testing

### Model Training Errors

If ARIMA fails to converge:
- Try different order parameters
- Ensure you have enough data points
- Check for missing values in your data

## Next Steps

1. Experiment with different model parameters
2. Try adding external features (e.g., economic indicators)
3. Implement cross-validation for time series
4. Explore deep learning models (LSTM, GRU)
5. Create ensemble models combining multiple approaches

## Resources

- [statsmodels documentation](https://www.statsmodels.org/)
- [scikit-learn documentation](https://scikit-learn.org/)
- [Time Series Analysis with Python](https://www.python.org/)

## Need Help?

- Check the Jupyter notebook for detailed examples
- Review the test scripts for working code samples
- Open an issue on GitHub for bugs or questions

Happy forecasting! 📈
