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
    
    def train_model(self, epochs=50, batch_size=32, validation_split=0.1, verbose=1):
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

def display_stock_menu():
    """Display available stock options"""
    popular_stocks = {
        '1': ('AAPL', 'Apple Inc.'),
        '2': ('GOOGL', 'Alphabet Inc. (Google)'),
        '3': ('MSFT', 'Microsoft Corporation'),
        '4': ('AMZN', 'Amazon.com Inc.'),
        '5': ('TSLA', 'Tesla Inc.'),
        '6': ('NVDA', 'NVIDIA Corporation'),
        '7': ('META', 'Meta Platforms Inc. (Facebook)'),
        '8': ('NFLX', 'Netflix Inc.'),
        '9': ('BABA', 'Alibaba Group Holding'),
        '10': ('V', 'Visa Inc.'),
        '11': ('JPM', 'JPMorgan Chase & Co.'),
        '12': ('JNJ', 'Johnson & Johnson'),
        '13': ('WMT', 'Walmart Inc.'),
        '14': ('PG', 'Procter & Gamble Co.'),
        '15': ('UNH', 'UnitedHealth Group Inc.'),
        '16': ('HD', 'The Home Depot Inc.'),
        '17': ('BAC', 'Bank of America Corp.'),
        '18': ('MA', 'Mastercard Inc.'),
        '19': ('DIS', 'The Walt Disney Company'),
        '20': ('ADBE', 'Adobe Inc.')
    }
    
    print("\n" + "="*60)
    print("📈 STOCK PRICE PREDICTION USING NEURAL NETWORKS 📈")
    print("="*60)
    print("\n🔥 POPULAR STOCKS TO ANALYZE:")
    print("-" * 40)
    
    # Display in two columns
    for i in range(0, len(popular_stocks), 2):
        left_key = str(i+1)
        right_key = str(i+2) if i+2 <= len(popular_stocks) else None
        
        if left_key in popular_stocks:
            left_symbol, left_name = popular_stocks[left_key]
            left_display = f"{left_key:2}. {left_symbol:5} - {left_name}"
        else:
            left_display = ""
            
        if right_key and right_key in popular_stocks:
            right_symbol, right_name = popular_stocks[right_key]
            right_display = f"{right_key:2}. {right_symbol:5} - {right_name}"
        else:
            right_display = ""
        
        if len(left_display) > 0:
            print(f"{left_display:<45} {right_display}")
    
    print("\n" + "-" * 40)
    print("21. Enter custom stock symbol")
    print("22. Analyze multiple stocks")
    print("23. Exit")
    print("-" * 40)
    
    return popular_stocks

def get_user_choice():
    """Get user's stock selection"""
    popular_stocks = display_stock_menu()
    
    while True:
        try:
            choice = input("\n👉 Enter your choice (1-23): ").strip()
            
            if choice == '23':
                print("👋 Thank you for using Stock Price Predictor!")
                return None, None
            
            elif choice == '21':
                custom_symbol = input("📝 Enter stock symbol (e.g., AAPL, TSLA): ").strip().upper()
                if custom_symbol:
                    return custom_symbol, 'single'
                else:
                    print("❌ Please enter a valid stock symbol.")
                    continue
            
            elif choice == '22':
                return None, 'multiple'
            
            elif choice in popular_stocks:
                symbol, name = popular_stocks[choice]
                print(f"✅ Selected: {symbol} - {name}")
                return symbol, 'single'
            
            else:
                print("❌ Invalid choice. Please enter a number between 1-23.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return None, None
        except Exception as e:
            print(f"❌ Error: {e}. Please try again.")

def get_analysis_parameters():
    """Get analysis parameters from user"""
    print("\n⚙️  ANALYSIS PARAMETERS")
    print("-" * 30)
    
    # Get lookback days
    while True:
        try:
            lookback = input("📅 Lookback days (default 60, recommended 30-120): ").strip()
            lookback_days = int(lookback) if lookback else 60
            if 10 <= lookback_days <= 200:
                break
            else:
                print("❌ Please enter a value between 10-200.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Get epochs
    while True:
        try:
            epoch_input = input("🔄 Training epochs (default 30, more = better but slower): ").strip()
            epochs = int(epoch_input) if epoch_input else 30
            if 10 <= epochs <= 200:
                break
            else:
                print("❌ Please enter a value between 10-200.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    # Get time period
    periods = {'1': '1y', '2': '2y', '3': '5y', '4': 'max'}
    print("\n📊 Data period:")
    print("1. 1 year    2. 2 years    3. 5 years    4. Maximum available")
    
    while True:
        period_choice = input("Select period (default 2): ").strip() or '2'
        if period_choice in periods:
            period = periods[period_choice]
            break
        else:
            print("❌ Please enter 1, 2, 3, or 4.")
    
    return lookback_days, epochs, period

def analyze_multiple_stocks():
    """Analyze multiple stocks"""
    default_stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    print("\n📊 MULTIPLE STOCK ANALYSIS")
    print("-" * 30)
    print("Default stocks:", ', '.join(default_stocks))
    
    choice = input("Use default list? (y/n, default y): ").strip().lower()
    
    if choice in ['n', 'no']:
        custom_input = input("Enter stock symbols separated by commas (e.g., AAPL,TSLA,MSFT): ").strip()
        if custom_input:
            stocks = [s.strip().upper() for s in custom_input.split(',')]
        else:
            stocks = default_stocks
    else:
        stocks = default_stocks
    
    print(f"📈 Will analyze: {', '.join(stocks)}")
    
    # Get parameters
    lookback_days, epochs, period = get_analysis_parameters()
    
    # Analyze each stock
    results_summary = []
    
    for i, stock in enumerate(stocks, 1):
        print(f"\n{'='*20} ANALYZING {stock} ({i}/{len(stocks)}) {'='*20}")
        
        try:
            predictor = StockPredictor(symbol=stock, period=period)
            result = predictor.run_complete_analysis(stock, lookback_days=lookback_days, epochs=epochs)
            
            if result:
                results_summary.append({
                    'symbol': stock,
                    'accuracy': result['metrics']['accuracy'],
                    'rmse': result['metrics']['rmse'],
                    'future_price': result['future_predictions'][0][0]
                })
        except Exception as e:
            print(f"❌ Error analyzing {stock}: {e}")
            continue
    
    # Display summary
    if results_summary:
        print(f"\n{'='*60}")
        print("📊 ANALYSIS SUMMARY")
        print(f"{'='*60}")
        print(f"{'Stock':<8} {'Accuracy':<12} {'RMSE':<12} {'Next Day Pred':<15}")
        print("-" * 60)
        
        for result in sorted(results_summary, key=lambda x: x['accuracy'], reverse=True):
            print(f"{result['symbol']:<8} {result['accuracy']:<11.2f}% "
                  f"${result['rmse']:<11.2f} ${result['future_price']:<14.2f}")
    
    return results_summary

# Example usage and main execution
if __name__ == "__main__":
    while True:
        try:
            # Get user choice
            symbol, analysis_type = get_user_choice()
            
            if symbol is None and analysis_type is None:
                break
            
            elif analysis_type == 'multiple':
                analyze_multiple_stocks()
                
            elif analysis_type == 'single':
                # Get analysis parameters
                lookback_days, epochs, period = get_analysis_parameters()
                
                print(f"\n{'='*20} ANALYZING {symbol} {'='*20}")
                
                # Run analysis
                predictor = StockPredictor(symbol=symbol, period=period)
                results = predictor.run_complete_analysis(symbol, lookback_days=lookback_days, epochs=epochs)
                
                if results:
                    print(f"\n✅ Analysis completed for {symbol}!")
                    print(f"📊 Accuracy: {results['metrics']['accuracy']:.2f}%")
                    print(f"📈 Next day prediction: ${results['future_predictions'][0][0]:.2f}")
            
            # Ask if user wants to analyze another stock
            print("\n" + "="*60)
            continue_choice = input("🔄 Analyze another stock? (y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes']:
                print("👋 Thank you for using Stock Price Predictor!")
                break
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            continue