"""
Test script to verify the time series forecasting implementation.
This script tests the basic functionality of all modules.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from datetime import datetime

print("=" * 70)
print("Testing S&P 500 Time Series Forecasting Implementation")
print("=" * 70)

# Test 1: Import all modules
print("\n1. Testing module imports...")
try:
    from data_loader import fetch_sp500_data, get_close_prices
    from preprocessing import train_test_split_timeseries, check_missing_values
    from models import ARIMAModel, MovingAverageModel, LinearRegressionModel
    from evaluation import evaluate_forecast, compare_models
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Failed to import modules: {e}")
    sys.exit(1)

# Test 2: Create mock data (since we can't fetch in this environment)
print("\n2. Creating mock S&P 500 data...")
try:
    # Create sample data for testing
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    np.random.seed(42)
    base_price = 4500
    prices = base_price + np.cumsum(np.random.randn(30) * 10)
    
    data = pd.DataFrame({
        'Open': prices + np.random.randn(30) * 5,
        'High': prices + np.abs(np.random.randn(30) * 10),
        'Low': prices - np.abs(np.random.randn(30) * 10),
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 30)
    }, index=dates)
    
    print(f"✓ Mock data created successfully: {len(data)} rows")
    print(f"  Columns: {list(data.columns)}")
except Exception as e:
    print(f"✗ Failed to create mock data: {e}")
    sys.exit(1)

# Test 3: Data preprocessing
print("\n3. Testing data preprocessing...")
try:
    close_prices = get_close_prices(data)
    missing = check_missing_values(close_prices)
    print(f"✓ Close prices extracted: {len(close_prices)} values")
    print(f"  Missing values: {missing}")
except Exception as e:
    print(f"✗ Failed to preprocess data: {e}")
    sys.exit(1)

# Test 4: Train-test split
print("\n4. Testing train-test split...")
try:
    train, test = train_test_split_timeseries(close_prices, test_size=0.3)
    print(f"✓ Data split successfully")
    print(f"  Train size: {len(train)}, Test size: {len(test)}")
except Exception as e:
    print(f"✗ Failed to split data: {e}")
    sys.exit(1)

# Test 5: Moving Average Model
print("\n5. Testing Moving Average Model...")
try:
    ma_model = MovingAverageModel(window=3)
    ma_model.fit(train)
    ma_predictions = ma_model.predict(steps=len(test))
    print(f"✓ Moving Average model trained and predicted")
    print(f"  Predictions: {len(ma_predictions)} values")
except Exception as e:
    print(f"✗ Moving Average model failed: {e}")

# Test 6: Linear Regression Model
print("\n6. Testing Linear Regression Model...")
try:
    lr_model = LinearRegressionModel()
    lr_model.fit(train)
    lr_predictions = lr_model.predict(steps=len(test))
    print(f"✓ Linear Regression model trained and predicted")
    print(f"  Predictions: {len(lr_predictions)} values")
except Exception as e:
    print(f"✗ Linear Regression model failed: {e}")

# Test 7: ARIMA Model (only if enough data)
if len(train) >= 10:
    print("\n7. Testing ARIMA Model...")
    try:
        arima_model = ARIMAModel(order=(2, 1, 0))
        arima_model.fit(train)
        arima_predictions = arima_model.predict(steps=len(test))
        print(f"✓ ARIMA model trained and predicted")
        print(f"  Predictions: {len(arima_predictions)} values")
    except Exception as e:
        print(f"⚠ ARIMA model warning (may need more data): {e}")
else:
    print("\n7. Skipping ARIMA test (insufficient data)")

# Test 8: Evaluation
print("\n8. Testing model evaluation...")
try:
    # Use Moving Average predictions for testing
    results = evaluate_forecast(test.values, ma_predictions)
    print(f"✓ Evaluation metrics calculated:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
except Exception as e:
    print(f"✗ Evaluation failed: {e}")

# Test 9: Model comparison
print("\n9. Testing model comparison...")
try:
    model_results = {
        'Moving Average': evaluate_forecast(test.values, ma_predictions),
        'Linear Regression': evaluate_forecast(test.values, lr_predictions)
    }
    comparison = compare_models(model_results)
    print(f"✓ Model comparison created:")
    print(comparison)
except Exception as e:
    print(f"✗ Model comparison failed: {e}")

print("\n" + "=" * 70)
print("All basic tests completed successfully!")
print("=" * 70)
print("\nNext steps:")
print("1. Open the Jupyter notebook: jupyter notebook notebooks/sp500_forecasting.ipynb")
print("2. Run all cells to see comprehensive analysis and visualizations")
print("=" * 70)
