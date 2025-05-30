import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
import base64
import io
import gc
import os

# Configure TensorFlow for memory efficiency
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
tf.config.experimental.enable_memory_growth = True

warnings.filterwarnings('ignore')

class StockPredictorWeb:
    def __init__(self, symbol='AAPL', period='1y'):  # Reduced default period
        self.symbol = symbol
        self.period = period
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.data = None
        self.scaled_data = None
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        print(f"Fetching data for {self.symbol}...")
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            
            if self.data.empty:
                raise ValueError(f"No data found for symbol {self.symbol}")
            
            # Limit data size to prevent memory issues
            if len(self.data) > 500:  # Limit to 500 days max
                self.data = self.data.tail(500)
                
            print(f"Data fetched: {len(self.data)} records from {self.data.index[0].date()} to {self.data.index[-1].date()}")
            return self.data
        except Exception as e:
            print(f"Error fetching data: {e}")
            raise
    
    def preprocess_data(self, lookback_days=30):  # Reduced default lookback
        """Preprocess data for LSTM model"""
        try:
            # Use closing prices
            prices = self.data['Close'].values.reshape(-1, 1)
            
            # Scale the data
            self.scaled_data = self.scaler.fit_transform(prices)
            
            # Create sequences for LSTM
            X, y = [], []
            for i in range(lookback_days, len(self.scaled_data)):
                X.append(self.scaled_data[i-lookback_days:i, 0])
                y.append(self.scaled_data[i, 0])
            
            X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)  # Use float32 to save memory
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Split into train and test sets (80-20 split)
            split_idx = int(len(X) * 0.8)
            self.X_train, self.X_test = X[:split_idx], X[split_idx:]
            self.y_train, self.y_test = y[:split_idx], y[split_idx:]
            
            print(f"Training data shape: {self.X_train.shape}")
            print(f"Testing data shape: {self.X_test.shape}")
            
            return self.X_train, self.X_test, self.y_train, self.y_test
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            raise
    
    def build_model(self, units=25, dropout=0.2, learning_rate=0.001, reduced_model=False):
        """Build LSTM neural network model"""
        try:
            # Use smaller model for memory efficiency
            if reduced_model:
                units = 15
                dropout = 0.3
            
            self.model = Sequential([
                LSTM(units=units, return_sequences=True, input_shape=(self.X_train.shape[1], 1)),
                Dropout(dropout),
                
                LSTM(units=units//2, return_sequences=False),  # Reduced second layer
                Dropout(dropout),
                
                Dense(units=1)
            ])
            
            optimizer = Adam(learning_rate=learning_rate)
            self.model.compile(optimizer=optimizer, loss='mean_squared_error')
            
            print("Model architecture:")
            self.model.summary()
            return self.model
        except Exception as e:
            print(f"Error building model: {e}")
            raise
    
    def train_model(self, epochs=10, batch_size=16, validation_split=0.1, verbose=0):  # Reduced defaults
        """Train the neural network"""
        try:
            print("Training the model...")
            
            # Use callbacks to prevent overfitting and save memory
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=5, 
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5, 
                    patience=3, 
                    min_lr=0.0001
                )
            ]
            
            history = self.model.fit(
                self.X_train, self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=verbose,
                shuffle=False,
                callbacks=callbacks
            )
            
            return history
        except Exception as e:
            print(f"Error training model: {e}")
            raise
    
    def make_predictions(self):
        """Make predictions on test data"""
        try:
            print("Making predictions...")
            predictions = self.model.predict(self.X_test, verbose=0)
            
            # Inverse transform to get actual prices
            predictions = self.scaler.inverse_transform(predictions)
            actual_prices = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
            
            return predictions, actual_prices
        except Exception as e:
            print(f"Error making predictions: {e}")
            raise
    
    def evaluate_model(self, predictions, actual_prices):
        """Evaluate model performance"""
        try:
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
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return {'mse': 0, 'mae': 0, 'rmse': 0, 'accuracy': 0}
    
    def create_training_plot(self, history):
        """Create training history plot and return as base64 string"""
        try:
            plt.figure(figsize=(10, 4))  # Reduced figure size
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss', color='blue')
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
            plt.title(f'{self.symbol} - Model Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=80, bbox_inches='tight')  # Reduced DPI
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_base64
        except Exception as e:
            print(f"Error creating training plot: {e}")
            return ""
    
    def create_prediction_plot(self, predictions, actual_prices):
        """Create prediction results plot and return as base64 string"""
        try:
            plt.figure(figsize=(12, 6))  # Reduced figure size
            
            # Get the dates for the test period
            test_dates = self.data.index[-len(actual_prices):]
            
            plt.subplot(2, 1, 1)
            plt.plot(test_dates, actual_prices, label='Actual Prices', color='blue', linewidth=2)
            plt.plot(test_dates, predictions, label='Predicted Prices', color='red', linewidth=2, alpha=0.7)
            plt.title(f'{self.symbol} Stock Price Prediction Results')
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
            
            # Convert plot to base64 string
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=80, bbox_inches='tight')  # Reduced DPI
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_base64
        except Exception as e:
            print(f"Error creating prediction plot: {e}")
            return ""
    
    def predict_future(self, days=7):  # Reduced default days
        """Predict future stock prices"""
        try:
            print(f"Predicting next {days} days...")
            
            # Get the last sequence from the data (use minimum of available data or required lookback)
            lookback = min(60, len(self.scaled_data))
            last_sequence = self.scaled_data[-lookback:].reshape(1, lookback, 1)
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
            
            return future_dates, future_prices
        except Exception as e:
            print(f"Error predicting future: {e}")
            return pd.date_range(start=pd.Timestamp.now(), periods=days), np.zeros((days, 1))
    
    def create_future_plot(self, future_dates, future_prices):
        """Create future predictions plot and return as base64 string"""
        try:
            plt.figure(figsize=(12, 5))  # Reduced figure size
            
            # Plot recent historical data (last 30 days)
            recent_data = self.data['Close'][-30:]
            plt.plot(recent_data.index, recent_data.values, label='Historical Prices', color='blue', linewidth=2)
            
            # Plot future predictions
            plt.plot(future_dates, future_prices, label='Future Predictions', color='red', linewidth=2, linestyle='--')
            
            plt.title(f'{self.symbol} Future Price Predictions ({len(future_dates)} days)')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Convert plot to base64 string
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=80, bbox_inches='tight')  # Reduced DPI
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_base64
        except Exception as e:
            print(f"Error creating future plot: {e}")
            return ""
    
    def create_price_history_plot(self):
        """Create stock price history plot and return as base64 string"""
        try:
            plt.figure(figsize=(12, 6))  # Reduced figure size
            
            # Plot price history (simplified)
            plt.subplot(2, 1, 1)
            plt.plot(self.data.index, self.data['Close'], label='Close Price', color='blue', linewidth=1)
            plt.title(f'{self.symbol} Stock Price History')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot volume (simplified)
            plt.subplot(2, 1, 2)
            plt.plot(self.data.index, self.data['Volume'], alpha=0.7, color='purple', linewidth=1)
            plt.title(f'{self.symbol} Trading Volume')
            plt.xlabel('Date')
            plt.ylabel('Volume')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Convert plot to base64 string
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=80, bbox_inches='tight')  # Reduced DPI
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
            
            return img_base64
        except Exception as e:
            print(f"Error creating price history plot: {e}")
            return ""
    
    def run_complete_analysis_web(self, lookback_days=30, epochs=10, reduced_model=True):
        """Run complete stock prediction analysis for web"""
        print(f"=== Stock Price Prediction Analysis for {self.symbol} ===\n")
        
        try:
            # Fetch and preprocess data
            self.fetch_data()
            self.preprocess_data(lookback_days)
            
            # Create price history plot
            price_history_plot = self.create_price_history_plot()
            
            # Build and train model
            self.build_model(reduced_model=reduced_model)
            history = self.train_model(epochs=epochs, verbose=0)
            
            # Create training plot
            training_plot = self.create_training_plot(history)
            
            # Make predictions and evaluate
            predictions, actual_prices = self.make_predictions()
            metrics = self.evaluate_model(predictions, actual_prices)
            
            # Create prediction plot
            prediction_plot = self.create_prediction_plot(predictions, actual_prices)
            
            # Predict future prices (reduced to 7 days)
            future_dates, future_prices = self.predict_future(7)
            
            # Create future prediction plot
            future_plot = self.create_future_plot(future_dates, future_prices)
            
            print(f"\nNext 5 day predictions for {self.symbol}:")
            for i in range(min(5, len(future_prices))):
                print(f"{future_dates[i].strftime('%Y-%m-%d')}: ${future_prices[i][0]:.2f}")
            
            # Clean up memory
            gc.collect()
            
            return {
                'model': self.model,
                'predictions': predictions,
                'actual_prices': actual_prices,
                'metrics': metrics,
                'future_predictions': future_prices[:7],
                'future_dates': future_dates[:7],
                'plots': {
                    'price_history': price_history_plot,
                    'training': training_plot,
                    'predictions': prediction_plot,
                    'future': future_plot
                }
            }
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            # Clean up TensorFlow session
            try:
                tf.keras.backend.clear_session()
            except:
                pass
            gc.collect()
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')  # Use non-interactive backend
# import yfinance as yf
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# import warnings
# import base64
# import io
# warnings.filterwarnings('ignore')

# class StockPredictorWeb:
#     def __init__(self, symbol='AAPL', period='2y'):
#         self.symbol = symbol
#         self.period = period
#         self.scaler = MinMaxScaler(feature_range=(0, 1))
#         self.model = None
#         self.data = None
#         self.scaled_data = None
        
#     def fetch_data(self):
#         """Fetch stock data from Yahoo Finance"""
#         print(f"Fetching data for {self.symbol}...")
#         ticker = yf.Ticker(self.symbol)
#         self.data = ticker.history(period=self.period)
        
#         if self.data.empty:
#             raise ValueError(f"No data found for symbol {self.symbol}")
            
#         print(f"Data fetched: {len(self.data)} records from {self.data.index[0].date()} to {self.data.index[-1].date()}")
#         return self.data
    
#     def preprocess_data(self, lookback_days=60):
#         """Preprocess data for LSTM model"""
#         # Use closing prices
#         prices = self.data['Close'].values.reshape(-1, 1)
        
#         # Scale the data
#         self.scaled_data = self.scaler.fit_transform(prices)
        
#         # Create sequences for LSTM
#         X, y = [], []
#         for i in range(lookback_days, len(self.scaled_data)):
#             X.append(self.scaled_data[i-lookback_days:i, 0])
#             y.append(self.scaled_data[i, 0])
        
#         X, y = np.array(X), np.array(y)
#         X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
#         # Split into train and test sets (80-20 split)
#         split_idx = int(len(X) * 0.8)
#         self.X_train, self.X_test = X[:split_idx], X[split_idx:]
#         self.y_train, self.y_test = y[:split_idx], y[split_idx:]
        
#         print(f"Training data shape: {self.X_train.shape}")
#         print(f"Testing data shape: {self.X_test.shape}")
        
#         return self.X_train, self.X_test, self.y_train, self.y_test
    
#     def build_model(self, units=50, dropout=0.2, learning_rate=0.001):
#         """Build LSTM neural network model"""
#         self.model = Sequential([
#             LSTM(units=units, return_sequences=True, input_shape=(self.X_train.shape[1], 1)),
#             Dropout(dropout),
            
#             LSTM(units=units, return_sequences=True),
#             Dropout(dropout),
            
#             LSTM(units=units),
#             Dropout(dropout),
            
#             Dense(units=1)
#         ])
        
#         optimizer = Adam(learning_rate=learning_rate)
#         self.model.compile(optimizer=optimizer, loss='mean_squared_error')
        
#         print("Model architecture:")
#         self.model.summary()
#         return self.model
    
#     def train_model(self, epochs=50, batch_size=32, validation_split=0.1, verbose=1):
#         """Train the neural network"""
#         print("Training the model...")
#         history = self.model.fit(
#             self.X_train, self.y_train,
#             epochs=epochs,
#             batch_size=batch_size,
#             validation_split=validation_split,
#             verbose=verbose,
#             shuffle=False
#         )
        
#         return history
    
#     def make_predictions(self):
#         """Make predictions on test data"""
#         print("Making predictions...")
#         predictions = self.model.predict(self.X_test)
        
#         # Inverse transform to get actual prices
#         predictions = self.scaler.inverse_transform(predictions)
#         actual_prices = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        
#         return predictions, actual_prices
    
#     def evaluate_model(self, predictions, actual_prices):
#         """Evaluate model performance"""
#         mse = mean_squared_error(actual_prices, predictions)
#         mae = mean_absolute_error(actual_prices, predictions)
#         rmse = np.sqrt(mse)
        
#         # Calculate accuracy as percentage of predictions within 5% of actual
#         accuracy = np.mean(np.abs((predictions - actual_prices) / actual_prices) < 0.05) * 100
        
#         print(f"\nModel Performance:")
#         print(f"Mean Squared Error: {mse:.4f}")
#         print(f"Mean Absolute Error: {mae:.4f}")
#         print(f"Root Mean Squared Error: {rmse:.4f}")
#         print(f"Accuracy (within 5%): {accuracy:.2f}%")
        
#         return {'mse': mse, 'mae': mae, 'rmse': rmse, 'accuracy': accuracy}
    
#     def create_training_plot(self, history):
#         """Create training history plot and return as base64 string"""
#         plt.figure(figsize=(12, 4))
        
#         plt.subplot(1, 2, 1)
#         plt.plot(history.history['loss'], label='Training Loss', color='blue')
#         plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
#         plt.title(f'{self.symbol} - Model Training Loss')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
        
#         plt.tight_layout()
        
#         # Convert plot to base64 string
#         img_buffer = io.BytesIO()
#         plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
#         img_buffer.seek(0)
#         img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
#         plt.close()
        
#         return img_base64
    
#     def create_prediction_plot(self, predictions, actual_prices):
#         """Create prediction results plot and return as base64 string"""
#         plt.figure(figsize=(15, 8))
        
#         # Get the dates for the test period
#         test_dates = self.data.index[-len(actual_prices):]
        
#         plt.subplot(2, 1, 1)
#         plt.plot(test_dates, actual_prices, label='Actual Prices', color='blue', linewidth=2)
#         plt.plot(test_dates, predictions, label='Predicted Prices', color='red', linewidth=2, alpha=0.7)
#         plt.title(f'{self.symbol} Stock Price Prediction Results')
#         plt.xlabel('Date')
#         plt.ylabel('Price ($)')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
        
#         # Plot the difference
#         plt.subplot(2, 1, 2)
#         difference = predictions.flatten() - actual_prices.flatten()
#         plt.plot(test_dates, difference, color='green', alpha=0.7)
#         plt.title('Prediction Error (Predicted - Actual)')
#         plt.xlabel('Date')
#         plt.ylabel('Error ($)')
#         plt.grid(True, alpha=0.3)
#         plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
#         plt.tight_layout()
        
#         # Convert plot to base64 string
#         img_buffer = io.BytesIO()
#         plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
#         img_buffer.seek(0)
#         img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
#         plt.close()
        
#         return img_base64
    
#     def predict_future(self, days=30):
#         """Predict future stock prices"""
#         print(f"Predicting next {days} days...")
        
#         # Get the last sequence from the data
#         last_sequence = self.scaled_data[-60:].reshape(1, 60, 1)
#         future_predictions = []
        
#         for _ in range(days):
#             # Predict next day
#             next_pred = self.model.predict(last_sequence, verbose=0)
#             future_predictions.append(next_pred[0, 0])
            
#             # Update sequence for next prediction
#             last_sequence = np.roll(last_sequence, -1, axis=1)
#             last_sequence[0, -1, 0] = next_pred[0, 0]
        
#         # Inverse transform predictions
#         future_predictions = np.array(future_predictions).reshape(-1, 1)
#         future_prices = self.scaler.inverse_transform(future_predictions)
        
#         # Create future dates
#         last_date = self.data.index[-1]
#         future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
        
#         return future_dates, future_prices
    
#     def create_future_plot(self, future_dates, future_prices):
#         """Create future predictions plot and return as base64 string"""
#         plt.figure(figsize=(15, 6))
        
#         # Plot recent historical data
#         recent_data = self.data['Close'][-60:]
#         plt.plot(recent_data.index, recent_data.values, label='Historical Prices', color='blue', linewidth=2)
        
#         # Plot future predictions
#         plt.plot(future_dates, future_prices, label='Future Predictions', color='red', linewidth=2, linestyle='--')
        
#         plt.title(f'{self.symbol} Future Price Predictions ({len(future_dates)} days)')
#         plt.xlabel('Date')
#         plt.ylabel('Price ($)')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
#         plt.xticks(rotation=45)
#         plt.tight_layout()
        
#         # Convert plot to base64 string
#         img_buffer = io.BytesIO()
#         plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
#         img_buffer.seek(0)
#         img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
#         plt.close()
        
#         return img_base64
    
#     def create_price_history_plot(self):
#         """Create stock price history plot and return as base64 string"""
#         plt.figure(figsize=(15, 8))
        
#         # Plot full price history
#         plt.subplot(2, 1, 1)
#         plt.plot(self.data.index, self.data['Close'], label='Close Price', color='blue', linewidth=1)
#         plt.plot(self.data.index, self.data['High'], label='High', color='green', alpha=0.5, linewidth=0.5)
#         plt.plot(self.data.index, self.data['Low'], label='Low', color='red', alpha=0.5, linewidth=0.5)
#         plt.title(f'{self.symbol} Stock Price History')
#         plt.xlabel('Date')
#         plt.ylabel('Price ($)')
#         plt.legend()
#         plt.grid(True, alpha=0.3)
        
#         # Plot volume
#         plt.subplot(2, 1, 2)
#         plt.bar(self.data.index, self.data['Volume'], alpha=0.7, color='purple')
#         plt.title(f'{self.symbol} Trading Volume')
#         plt.xlabel('Date')
#         plt.ylabel('Volume')
#         plt.grid(True, alpha=0.3)
        
#         plt.tight_layout()
        
#         # Convert plot to base64 string
#         img_buffer = io.BytesIO()
#         plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
#         img_buffer.seek(0)
#         img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
#         plt.close()
        
#         return img_base64
    
#     def run_complete_analysis_web(self, lookback_days=60, epochs=50):
#         """Run complete stock prediction analysis for web"""
#         print(f"=== Stock Price Prediction Analysis for {self.symbol} ===\n")
        
#         try:
#             # Fetch and preprocess data
#             self.fetch_data()
#             self.preprocess_data(lookback_days)
            
#             # Create price history plot
#             price_history_plot = self.create_price_history_plot()
            
#             # Build and train model
#             self.build_model()
#             history = self.train_model(epochs=epochs, verbose=0)
            
#             # Create training plot
#             training_plot = self.create_training_plot(history)
            
#             # Make predictions and evaluate
#             predictions, actual_prices = self.make_predictions()
#             metrics = self.evaluate_model(predictions, actual_prices)
            
#             # Create prediction plot
#             prediction_plot = self.create_prediction_plot(predictions, actual_prices)
            
#             # Predict future prices
#             future_dates, future_prices = self.predict_future(30)
            
#             # Create future prediction plot
#             future_plot = self.create_future_plot(future_dates, future_prices)
            
#             print(f"\nNext 5 day predictions for {self.symbol}:")
#             for i in range(5):
#                 print(f"{future_dates[i].strftime('%Y-%m-%d')}: ${future_prices[i][0]:.2f}")
            
#             return {
#                 'model': self.model,
#                 'predictions': predictions,
#                 'actual_prices': actual_prices,
#                 'metrics': metrics,
#                 'future_predictions': future_prices[:7],  # Return 7 days for web display
#                 'future_dates': future_dates[:7],
#                 'plots': {
#                     'price_history': price_history_plot,
#                     'training': training_plot,
#                     'predictions': prediction_plot,
#                     'future': future_plot
#                 }
#             }
            
#         except Exception as e:
#             print(f"Error in analysis: {str(e)}")
#             return None
