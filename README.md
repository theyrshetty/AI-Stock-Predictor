# AI Stock Predictor
## Stock Price Prediction with LSTM Neural Networks

A comprehensive web application for predicting stock prices using LSTM (Long Short-Term Memory) neural networks. This project fetches real-time stock data, trains deep learning models, and provides both historical analysis and future price predictions with interactive visualizations.

## Features

- **Real-time Data Fetching**: Automatically retrieves stock data from Yahoo Finance
- **LSTM Neural Network**: Deep learning model specifically designed for time series prediction
- **Interactive Web Interface**: User-friendly web application for stock analysis
- **Multiple Visualizations**: 
  - Stock price history with trading volume
  - Model training progress
  - Prediction accuracy comparison
  - Future price predictions
- **Performance Metrics**: Comprehensive model evaluation with MSE, MAE, RMSE, and accuracy
- **Future Predictions**: Predict stock prices for the next 30 days
- **Customizable Parameters**: Adjustable lookback periods, epochs, and model architecture

## Project Structure

```
stock_prediction/
├── __pycache__/           # Python cache files
├── static/               # Static web assets
│   └── plots/           # Generated plot images
├── templates/           # HTML templates
│   ├── about.html       # About page
│   └── index.html       # Main application interface
├── app.py              # Flask web application
├── spp2.py             # Main stock prediction class
└── README.md           # Project documentation
```

## Requirements

### Dependencies

```python
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
yfinance>=0.1.70
scikit-learn>=1.0.0
tensorflow>=2.8.0
flask>=2.0.0
```

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/theyrshetty/AI-Stock-Predictor
   cd stock_prediction
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Web Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Open your browser** and navigate to `http://localhost:5000`

3. **Enter a stock symbol** (e.g., AAPL, GOOGL, TSLA) and start the analysis

### Direct Python Usage

```python
from spp2 import StockPredictorWeb

# Initialize the predictor
predictor = StockPredictorWeb(symbol='AAPL', period='2y')

# Run complete analysis
results = predictor.run_complete_analysis_web(
    lookback_days=60, 
    epochs=50
)

# Access results
if results:
    print(f"Model Accuracy: {results['metrics']['accuracy']:.2f}%")
    print(f"RMSE: {results['metrics']['rmse']:.4f}")
```

## Model Architecture

The LSTM model consists of:

- **3 LSTM layers** with 50 units each
- **Dropout layers** (0.2 rate) for regularization
- **Dense output layer** for final prediction
- **Adam optimizer** with configurable learning rate
- **Mean Squared Error** loss function

### Default Parameters

- **Lookback Period**: 60 days
- **Training Epochs**: 50
- **Batch Size**: 32
- **Train/Test Split**: 80/20
- **Validation Split**: 10%
- **Dropout Rate**: 0.2
- **Learning Rate**: 0.001

## Features Explained

### Data Processing

1. **Data Fetching**: Retrieves historical stock data using Yahoo Finance API
2. **Normalization**: Scales data to 0-1 range using MinMaxScaler
3. **Sequence Creation**: Creates time series sequences for LSTM input
4. **Train/Test Split**: Chronological split to prevent data leakage

### Predictions

- **Historical Backtesting**: Tests model performance on unseen historical data
- **Future Predictions**: Predicts next 30 days using recursive forecasting
- **Confidence Metrics**: Provides accuracy within 5% threshold

### Visualizations

1. **Price History**: Shows historical prices with high/low ranges and trading volume
2. **Training Progress**: Displays loss curves for training and validation
3. **Prediction Comparison**: Compares actual vs predicted prices with error analysis
4. **Future Forecast**: Shows predicted future prices with historical context

## Performance Metrics

- **MSE (Mean Squared Error)**: Average squared difference between actual and predicted values
- **MAE (Mean Absolute Error)**: Average absolute difference
- **RMSE (Root Mean Squared Error)**: Square root of MSE for same-unit comparison
- **Accuracy**: Percentage of predictions within 5% of actual values

## Supported Stock Symbols

The application supports any stock symbol available on Yahoo Finance, including:

- **US Stocks**: AAPL, GOOGL, MSFT, TSLA, AMZN, etc.
- **International Stocks**: Use appropriate suffixes (e.g., .TO for Toronto)
- **ETFs**: SPY, QQQ, VTI, etc.
- **Indices**: ^GSPC (S&P 500), ^IXIC (NASDAQ), etc.

## Technical Details

### Data Requirements

- **Minimum Period**: 2 years of historical data recommended
- **Lookback Window**: 60 days used for prediction (configurable)
- **Update Frequency**: Real-time data fetching on each analysis

### Model Training

- **Non-interactive Backend**: Uses 'Agg' backend for server environments
- **Memory Efficient**: Processes data in batches
- **Reproducible**: Consistent results with fixed random seeds

## Limitations

- **Market Volatility**: Predictions may be less accurate during high volatility periods
- **External Factors**: Model doesn't account for news, earnings, or market sentiment
- **Short-term Focus**: Optimized for short to medium-term predictions
- **Data Dependency**: Requires sufficient historical data for training

## Future Enhancements

- **Multi-feature Input**: Incorporate volume, technical indicators
- **Ensemble Models**: Combine multiple prediction models
- **Real-time Updates**: Live prediction updates
- **Portfolio Analysis**: Multi-stock prediction and optimization
- **API Integration**: RESTful API for external integrations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

⚠️ **Investment Warning**: This tool is for educational and research purposes only. Stock market predictions are inherently uncertain, and past performance does not guarantee future results. Always conduct your own research and consider consulting with financial advisors before making investment decisions.

## Support

For questions, issues, or contributions:

- **GitHub Issues**: Report bugs or request features
- **Documentation**: Check inline code comments for detailed explanations
- **Community**: Join discussions in the project's discussion forum

---
