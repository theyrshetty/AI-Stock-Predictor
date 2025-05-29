import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    def __init__(self, symbol='AAPL', period='2y'):
        self.symbol = symbol
        self.period = period
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.data = None
        self.scaled_data = None
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        print(f"Fetching data for {self.symbol}...")
        ticker = yf.Ticker(self.symbol)
        self.data = ticker.history(period=self.period)
        
        if self.data.empty:
            raise ValueError(f"No data found for symbol {self.symbol}")
            
        print(f"Data fetched: {len(self.data)} records from {self.data.index[0].date()} to {self.data.index[-1].date()}")
        return self.data
    
    def preprocess_data(self, lookback_days=60):
        """Preprocess data for LSTM model"""
        # Use closing prices
        prices = self.data['Close'].values.reshape(-1, 1)
        
        # Scale the data
        self.scaled_data = self.scaler.fit_transform(prices)
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(lookback_days, len(self.scaled_data)):
            X.append(self.scaled_data[i-lookback_days:i, 0])
            y.append(self.scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split into train and test sets (80-20 split)
        split_idx = int(len(X) * 0.8)
        self.X_train, self.X_test = X[:split_idx], X[split_idx:]
        self.y_train, self.y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Testing data shape: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def build_model(self, units=50, dropout=0.2, learning_rate=0.001):
        """Build LSTM neural network model"""
        self.model = Sequential([
            LSTM(units=units, return_sequences=True, input_shape=(self.X_train.shape[1], 1)),
            Dropout(dropout),
            
            LSTM(units=units, return_sequences=True),
            Dropout(dropout),
            
            LSTM(units=units),
            Dropout(dropout),
            
            Dense(units=1)
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        print("Model architecture:")
        self.model.summary()
        return self.model
    
    def train_model(self, epochs=50, batch_size=32, validation_split=0.1):
        """Train the neural network"""
        print("Training the model...")
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1,
            shuffle=False
        )
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        return history
    
    def make_predictions(self):
        """Make predictions on test data"""
        print("Making predictions...")
        predictions = self.model.predict(self.X_test)
        
        # Inverse transform to get actual prices
        predictions = self.scaler.inverse_transform(predictions)
        actual_prices = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        
        return predictions, actual_prices
    
    def evaluate_model(self, predictions, actual_prices):
        """Evaluate model performance"""
        mse = mean_squared_error(actual_prices, predictions)
        mae = mean_absolute_error(actual_prices, predictions)
        rmse = np.sqrt(mse)
        
        # Calculate accuracy as percentage of predictions within 5% of actual
        accuracy = np.mean(np.abs((predictions - actual_prices) / actual_prices) < 0.05) * 100
        
        print(f"\nModel Performance:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Root Mean Squared Error: {rmse:.4f}")
        print(f"Accuracy (within 5%): {accuracy:.2f}%")
        
        return {'mse': mse, 'mae': mae, 'rmse': rmse, 'accuracy': accuracy}
    
    def plot_results(self, predictions, actual_prices):
        """Plot actual vs predicted prices"""
        plt.figure(figsize=(15, 8))
        
        # Get the dates for the test period
        test_dates = self.data.index[-len(actual_prices):]
        
        plt.subplot(2, 1, 1)
        plt.plot(test_dates, actual_prices, label='Actual Prices', color='blue', linewidth=2)
        plt.plot(test_dates, predictions, label='Predicted Prices', color='red', linewidth=2, alpha=0.7)
        plt.title(f'{self.symbol} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot the difference
        plt.subplot(2, 1, 2)
        difference = predictions.flatten() - actual_prices.flatten()
        plt.plot(test_dates, difference, color='green', alpha=0.7)
        plt.title('Prediction Error (Predicted - Actual)')
        plt.xlabel('Date')
        plt.ylabel('Error ($)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def predict_future(self, days=30):
        """Predict future stock prices"""
        print(f"Predicting next {days} days...")
        
        # Get the last sequence from the data
        last_sequence = self.scaled_data[-60:].reshape(1, 60, 1)
        future_predictions = []
        
        for _ in range(days):
            # Predict next day
            next_pred = self.model.predict(last_sequence, verbose=0)
            future_predictions.append(next_pred[0, 0])
            
            # Update sequence for next prediction
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = next_pred[0, 0]
        
        # Inverse transform predictions
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_prices = self.scaler.inverse_transform(future_predictions)
        
        # Create future dates
        last_date = self.data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
        
        # Plot future predictions
        plt.figure(figsize=(15, 6))
        
        # Plot recent historical data
        recent_data = self.data['Close'][-60:]
        plt.plot(recent_data.index, recent_data.values, label='Historical Prices', color='blue', linewidth=2)
        
        # Plot future predictions
        plt.plot(future_dates, future_prices, label='Future Predictions', color='red', linewidth=2, linestyle='--')
        
        plt.title(f'{self.symbol} Future Price Predictions ({days} days)')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return future_dates, future_prices
    
    def run_complete_analysis(self, symbol='AAPL', lookback_days=60, epochs=50):
        """Run complete stock prediction analysis"""
        print(f"=== Stock Price Prediction Analysis for {symbol} ===\n")
        
        try:
            # Fetch and preprocess data
            self.symbol = symbol
            self.fetch_data()
            self.preprocess_data(lookback_days)
            
            # Build and train model
            self.build_model()
            self.train_model(epochs=epochs)
            
            # Make predictions and evaluate
            predictions, actual_prices = self.make_predictions()
            metrics = self.evaluate_model(predictions, actual_prices)
            
            # Plot results
            self.plot_results(predictions, actual_prices)
            
            # Predict future prices
            future_dates, future_prices = self.predict_future(30)
            
            print(f"\nNext 5 day predictions for {symbol}:")
            for i in range(5):
                print(f"{future_dates[i].strftime('%Y-%m-%d')}: ${future_prices[i][0]:.2f}")
            
            return {
                'model': self.model,
                'predictions': predictions,
                'actual_prices': actual_prices,
                'metrics': metrics,
                'future_predictions': future_prices[:5]
            }
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return None

# Example usage and main execution
if __name__ == "__main__":
    # Initialize predictor
    predictor = StockPredictor()
    
    # Run analysis for different stocks
    stocks_to_analyze = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    print("Stock Price Prediction using Neural Networks")
    print("=" * 50)
    
    # You can run analysis for a single stock
    results = predictor.run_complete_analysis('AAPL', lookback_days=60, epochs=30)
    
    # Or analyze multiple stocks (uncomment below)
    """
    results_all = {}
    for stock in stocks_to_analyze:
        print(f"\n\nAnalyzing {stock}...")
        predictor_stock = StockPredictor()
        results_all[stock] = predictor_stock.run_complete_analysis(stock, epochs=30)
    """ 