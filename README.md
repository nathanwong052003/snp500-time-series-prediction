# S&P 500 Time Series Prediction

This project demonstrates various time series forecasting methods applied to S&P 500 stock prices.

## Overview

The project includes implementations of multiple time series forecasting models to predict S&P 500 stock prices:
- **Moving Average**
- **Linear Regression**
- **ARIMA (AutoRegressive Integrated Moving Average)**
- **SARIMAX (Seasonal ARIMA with eXogenous factors)**
- **Exponential Smoothing**

## Project Structure

```
snp500-time-series-prediction/
├── data/                    # Data directory for storing S&P 500 data
├── notebooks/               # Jupyter notebooks
│   └── sp500_forecasting.ipynb  # Main forecasting notebook
├── src/                     # Source code
│   ├── data_loader.py       # Data fetching and loading utilities
│   ├── preprocessing.py     # Data preprocessing functions
│   ├── models.py            # Time series model implementations
│   └── evaluation.py        # Model evaluation metrics
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nathanwong052003/snp500-time-series-prediction.git
cd snp500-time-series-prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Using Jupyter Notebook

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `notebooks/sp500_forecasting.ipynb`

3. Run the cells sequentially to:
   - Fetch S&P 500 data
   - Perform exploratory data analysis
   - Train multiple forecasting models
   - Compare model performance
   - Generate future predictions

### Using Python Files

You can also use the individual Python modules:

```python
from src.data_loader import fetch_sp500_data
from src.preprocessing import train_test_split_timeseries
from src.models import ARIMAModel
from src.evaluation import evaluate_forecast

# Fetch data
data = fetch_sp500_data(period="5y")

# Preprocess
train, test = train_test_split_timeseries(data['Close'])

# Train model
model = ARIMAModel(order=(5, 1, 0))
model.fit(train)

# Predict
predictions = model.predict(steps=len(test))

# Evaluate
results = evaluate_forecast(test.values, predictions)
print(results)
```

## Features

### Data Loading (`data_loader.py`)
- Fetch real-time S&P 500 data from Yahoo Finance
- Load/save data from/to CSV files
- Extract close prices for analysis

### Preprocessing (`preprocessing.py`)
- Handle missing values
- Create lag features
- Create rolling window features
- Normalize data
- Train-test split for time series

### Models (`models.py`)
- **ARIMAModel**: Classic statistical model for time series
- **SARIMAXModel**: ARIMA with seasonal components
- **ExponentialSmoothingModel**: Weighted average model
- **MovingAverageModel**: Simple moving average
- **LinearRegressionModel**: Trend-based linear model

### Evaluation (`evaluation.py`)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- R² (Coefficient of Determination)
- Model comparison utilities

## Results

The notebook provides comprehensive analysis including:
- Historical price visualization
- Daily returns analysis
- Model performance comparison
- Future price predictions
- Evaluation metrics for all models

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- yfinance
- prophet
- jupyter

See `requirements.txt` for specific versions.

## Disclaimer

**This project is for educational and research purposes only.** The predictions and analysis provided should not be used as the sole basis for investment decisions. Stock market predictions are inherently uncertain, and past performance does not guarantee future results.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Future Enhancements

- [ ] Add deep learning models (LSTM, GRU, Transformer)
- [ ] Incorporate sentiment analysis from news
- [ ] Add technical indicators as features
- [ ] Implement ensemble methods
- [ ] Create interactive dashboard with Plotly/Dash
- [ ] Add automated hyperparameter tuning
- [ ] Include confidence intervals for predictions

## Contact

For questions or suggestions, please open an issue on GitHub.