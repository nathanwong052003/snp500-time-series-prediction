"""
Example usage of the S&P 500 time series forecasting package.

This script demonstrates how to use the various modules to:
1. Fetch S&P 500 data
2. Preprocess the data
3. Train forecasting models
4. Make predictions
5. Evaluate model performance
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import our custom modules
from data_loader import fetch_sp500_data, save_data, get_close_prices
from preprocessing import train_test_split_timeseries, fill_missing_values
from models import ARIMAModel, LinearRegressionModel, MovingAverageModel
from evaluation import evaluate_forecast, print_evaluation_results, compare_models

def main():
    print("=" * 70)
    print("S&P 500 Time Series Forecasting - Example Usage")
    print("=" * 70)
    
    # Step 1: Fetch data
    print("\n[Step 1] Fetching S&P 500 data...")
    try:
        # For this example, we'll use mock data since we can't access internet
        # In real usage, uncomment the line below:
        # data = fetch_sp500_data(period="2y")
        
        # Create mock data for demonstration
        dates = pd.date_range(start='2022-01-01', periods=500, freq='D')
        np.random.seed(42)
        base_price = 4000
        prices = base_price + np.cumsum(np.random.randn(500) * 10)
        
        data = pd.DataFrame({
            'Open': prices + np.random.randn(500) * 5,
            'High': prices + np.abs(np.random.randn(500) * 10),
            'Low': prices - np.abs(np.random.randn(500) * 10),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 500)
        }, index=dates)
        
        print(f"✓ Data loaded: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
    except Exception as e:
        print(f"✗ Failed to fetch data: {e}")
        return
    
    # Step 2: Extract and preprocess close prices
    print("\n[Step 2] Preprocessing data...")
    close_prices = get_close_prices(data)
    
    # Handle missing values if any
    if close_prices.isnull().sum() > 0:
        close_prices = fill_missing_values(close_prices)
        print(f"✓ Filled missing values")
    
    # Split into train and test sets
    train_data, test_data = train_test_split_timeseries(close_prices, test_size=0.2)
    print(f"✓ Train set: {len(train_data)} samples")
    print(f"✓ Test set: {len(test_data)} samples")
    
    # Step 3: Train models
    print("\n[Step 3] Training models...")
    
    # Model 1: Moving Average
    print("\nTraining Moving Average model...")
    ma_model = MovingAverageModel(window=7)
    ma_model.fit(train_data)
    ma_predictions = ma_model.predict(steps=len(test_data))
    print("✓ Moving Average model trained")
    
    # Model 2: Linear Regression
    print("\nTraining Linear Regression model...")
    lr_model = LinearRegressionModel()
    lr_model.fit(train_data)
    lr_predictions = lr_model.predict(steps=len(test_data))
    print("✓ Linear Regression model trained")
    
    # Model 3: ARIMA
    print("\nTraining ARIMA model...")
    arima_model = ARIMAModel(order=(5, 1, 0))
    arima_model.fit(train_data)
    arima_predictions = arima_model.predict(steps=len(test_data))
    print("✓ ARIMA model trained")
    
    # Step 4: Evaluate models
    print("\n[Step 4] Evaluating models...")
    
    ma_results = evaluate_forecast(test_data.values, ma_predictions)
    lr_results = evaluate_forecast(test_data.values, lr_predictions)
    arima_results = evaluate_forecast(test_data.values, arima_predictions)
    
    print("\n--- Moving Average Results ---")
    print_evaluation_results(ma_results)
    
    print("\n--- Linear Regression Results ---")
    print_evaluation_results(lr_results)
    
    print("\n--- ARIMA Results ---")
    print_evaluation_results(arima_results)
    
    # Step 5: Compare models
    print("\n[Step 5] Comparing all models...")
    all_results = {
        'Moving Average': ma_results,
        'Linear Regression': lr_results,
        'ARIMA': arima_results
    }
    
    comparison = compare_models(all_results)
    print("\nModel Comparison:")
    print(comparison)
    
    # Find best model
    best_model = comparison['RMSE'].idxmin()
    print(f"\n🏆 Best model based on RMSE: {best_model}")
    print(f"   RMSE: {comparison.loc[best_model, 'RMSE']:.4f}")
    
    # Step 6: Visualize results
    print("\n[Step 6] Generating visualization...")
    
    plt.figure(figsize=(15, 8))
    
    # Plot actual data
    plt.plot(train_data.index, train_data.values, label='Training Data', linewidth=1.5, alpha=0.7)
    plt.plot(test_data.index, test_data.values, label='Actual Test Data', linewidth=2, color='black')
    
    # Plot predictions
    plt.plot(test_data.index, ma_predictions, label='Moving Average', linewidth=1.5, linestyle='--')
    plt.plot(test_data.index, lr_predictions, label='Linear Regression', linewidth=1.5, linestyle='--')
    plt.plot(test_data.index, arima_predictions, label='ARIMA', linewidth=1.5, linestyle='--')
    
    plt.title('S&P 500 Price Predictions Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Close Price (USD)', fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    output_file = 'predictions_comparison.png'
    plt.savefig(output_file, dpi=150)
    print(f"✓ Visualization saved to: {output_file}")
    
    # Step 7: Make future predictions
    print("\n[Step 7] Making future predictions...")
    
    # Retrain best model on full dataset
    if best_model == 'ARIMA':
        future_model = ARIMAModel(order=(5, 1, 0))
    elif best_model == 'Linear Regression':
        future_model = LinearRegressionModel()
    else:
        future_model = MovingAverageModel(window=7)
    
    future_model.fit(close_prices)
    future_predictions = future_model.predict(steps=30)
    
    print(f"✓ Generated predictions for next 30 days")
    print(f"  First predicted price: ${future_predictions[0]:.2f}")
    print(f"  Last predicted price: ${future_predictions[-1]:.2f}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print("\nFor more detailed analysis and visualizations, see:")
    print("  notebooks/sp500_forecasting.ipynb")
    print("=" * 70)

if __name__ == "__main__":
    main()
