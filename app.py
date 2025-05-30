from flask import Flask, render_template, request, jsonify, flash
import json
import os
from datetime import datetime, timedelta
import yfinance as yf
import traceback
import sys
import io
import gc
import psutil
import tensorflow as tf

# Configure TensorFlow to use CPU only and limit memory
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
tf.config.set_visible_devices([], 'GPU')  # Disable GPU

# Import the modified StockPredictor class
from spp2 import StockPredictorWeb

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-insecure-key')

# Popular stocks for quick selection
POPULAR_STOCKS = {
    'AAPL': 'Apple Inc.',
    'GOOGL': 'Alphabet Inc.',
    'MSFT': 'Microsoft Corp.',
    'AMZN': 'Amazon.com Inc.',
    'TSLA': 'Tesla Inc.',
    'NVDA': 'NVIDIA Corp.',
    'META': 'Meta Platforms',
    'NFLX': 'Netflix Inc.',
    'BABA': 'Alibaba Group',
    'V': 'Visa Inc.',
    'JPM': 'JPMorgan Chase',
    'JNJ': 'Johnson & Johnson',
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'Tata Consultancy Services',
    'INFY.NS': 'Infosys',
    'HDFCBANK.NS': 'HDFC Bank',
    'ICICIBANK.NS': 'ICICI Bank',
    'SBIN.NS': 'State Bank of India',
    'TITAN.NS': 'Titan Company Limited',
    'SAIL.NS': 'Steel Authority of India Limited',
}

def check_memory_usage():
    """Check current memory usage"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    return memory_mb

def validate_stock_symbol(symbol):
    """Validate if stock symbol exists"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='5d')
        return not hist.empty
    except Exception as e:
        print(f"Stock validation error: {e}")
        return False

def get_current_stock_price(symbol):
    """Get current stock price"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1d')
        if not hist.empty:
            return hist['Close'].iloc[-1]
    except Exception as e:
        print(f"Price fetch error: {e}")
        pass
    return None

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    
    if request.method == 'POST':
        predictor = None
        try:
            # Check initial memory
            initial_memory = check_memory_usage()
            print(f"Initial memory usage: {initial_memory:.2f} MB")
            
            # Get form data with more conservative defaults
            symbol = request.form['symbol'].upper().strip()
            lookback = min(int(request.form.get('lookback', 30)), 60)  # Max 60 days
            epochs = min(int(request.form.get('epochs', 10)), 20)     # Max 20 epochs
            period = request.form.get('period', '1y')  # Reduced to 1 year
            
            # Validate inputs
            if not symbol:
                flash('Please enter a stock symbol', 'error')
                return render_template("index.html", popular_stocks=POPULAR_STOCKS)
            
            if not (10 <= lookback <= 60):  # Reduced max lookback
                flash('Lookback days must be between 10 and 60', 'error')
                return render_template("index.html", popular_stocks=POPULAR_STOCKS)
            
            if not (5 <= epochs <= 20):  # Reduced max epochs
                flash('Epochs must be between 5 and 20', 'error')
                return render_template("index.html", popular_stocks=POPULAR_STOCKS)
            
            # Validate stock symbol
            if not validate_stock_symbol(symbol):
                flash(f'Invalid stock symbol: {symbol}', 'error')
                return render_template("index.html", popular_stocks=POPULAR_STOCKS)
            
            # Get current price for comparison
            current_price = get_current_stock_price(symbol)
            
            # Check memory before creating predictor
            current_memory = check_memory_usage()
            if current_memory > 400:  # If using more than 400MB
                gc.collect()  # Force garbage collection
                print(f"Memory after GC: {check_memory_usage():.2f} MB")
            
            # Initialize predictor with reduced memory footprint
            predictor = StockPredictorWeb(symbol=symbol, period=period)
            
            # Suppress console output during training
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            try:
                # Run complete analysis with reduced parameters
                analysis_result = predictor.run_complete_analysis_web(
                    lookback_days=lookback, 
                    epochs=epochs,
                    reduced_model=True  # Use smaller model
                )
            finally:
                sys.stdout = old_stdout
            
            if not analysis_result:
                flash(f'Analysis failed for {symbol}. Try with smaller parameters.', 'error')
                return render_template("index.html", popular_stocks=POPULAR_STOCKS)
            
            # Prepare result data
            result = {
                'symbol': symbol,
                'company_name': POPULAR_STOCKS.get(symbol, 'Unknown Company'),
                'current_price': current_price,
                'accuracy': f"{analysis_result['metrics']['accuracy']:.2f}",
                'rmse': f"{analysis_result['metrics']['rmse']:.2f}",
                'mae': f"{analysis_result['metrics']['mae']:.2f}",
                'predictions': [
                    {
                        'date': analysis_result['future_dates'][i].strftime('%Y-%m-%d'),
                        'price': f"{analysis_result['future_predictions'][i][0]:.2f}"
                    } for i in range(min(7, len(analysis_result['future_predictions'])))
                ],
                'data_range': f"{predictor.data.index[0].strftime('%Y-%m-%d')} to {predictor.data.index[-1].strftime('%Y-%m-%d')}",
                'training_params': {
                    'lookback_days': lookback,
                    'epochs': epochs,
                    'period': period,
                    'data_points': len(predictor.data)
                },
                'plots': analysis_result['plots']
            }
            
            # Calculate price change prediction
            if current_price and len(analysis_result['future_predictions']) > 0:
                predicted_price = analysis_result['future_predictions'][0][0]
                price_change = predicted_price - current_price
                price_change_pct = (price_change / current_price) * 100
                result['price_change'] = f"{price_change:+.2f}"
                result['price_change_pct'] = f"{price_change_pct:+.2f}"
                result['trend'] = 'up' if price_change > 0 else 'down' if price_change < 0 else 'flat'
            
            flash(f'Prediction completed for {symbol}!', 'success')
            
            # Log memory usage
            final_memory = check_memory_usage()
            print(f"Final memory usage: {final_memory:.2f} MB")
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error: {error_msg}")
            print(traceback.format_exc())
            
            # Check if it's a memory error
            if "memory" in error_msg.lower() or "killed" in error_msg.lower():
                flash('Prediction failed due to memory constraints. Try reducing lookback days or epochs.', 'error')
            else:
                flash(f'Error processing prediction: {error_msg}', 'error')
            result = None
        
        finally:
            # Clean up resources
            if predictor:
                del predictor
            gc.collect()  # Force garbage collection
            
            # Clear TensorFlow session
            try:
                tf.keras.backend.clear_session()
            except:
                pass
    
    return render_template("index.html", result=result, popular_stocks=POPULAR_STOCKS)

@app.route('/api/validate_symbol')
def validate_symbol():
    """API endpoint to validate stock symbol"""
    symbol = request.args.get('symbol', '').upper().strip()
    if not symbol:
        return jsonify({'valid': False, 'message': 'No symbol provided'})
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        hist = ticker.history(period='5d')
        
        if hist.empty:
            return jsonify({'valid': False, 'message': 'No data available'})
        
        return jsonify({
            'valid': True,
            'name': info.get('longName', 'Unknown'),
            'current_price': hist['Close'].iloc[-1]
        })
    except Exception as e:
        return jsonify({'valid': False, 'message': 'Invalid symbol'})

@app.route('/about')
def about():
    """About page explaining the model"""
    return render_template('about.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    memory_usage = check_memory_usage()
    return jsonify({
        'status': 'healthy',
        'memory_mb': memory_usage,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    os.makedirs('static/plots', exist_ok=True)
    
    # Configure for production
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"Starting app on port {port}, debug={debug}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    
    app.run(debug=debug, host='0.0.0.0', port=port)
# from flask import Flask, render_template, request, jsonify, flash
# import json
# import os
# from datetime import datetime, timedelta
# import yfinance as yf
# import traceback
# import sys
# import io

# # Import the modified StockPredictor class
# from spp2 import StockPredictorWeb

# app = Flask(__name__)
# app.secret_key = os.environ.get('SECRET_KEY', 'dev-insecure-key')

# # Popular stocks for quick selection
# POPULAR_STOCKS = {
#     'AAPL': 'Apple Inc.',
#     'GOOGL': 'Alphabet Inc.',
#     'MSFT': 'Microsoft Corp.',
#     'AMZN': 'Amazon.com Inc.',
#     'TSLA': 'Tesla Inc.',
#     'NVDA': 'NVIDIA Corp.',
#     'META': 'Meta Platforms',
#     'NFLX': 'Netflix Inc.',
#     'BABA': 'Alibaba Group',
#     'V': 'Visa Inc.',
#     'JPM': 'JPMorgan Chase',
#     'JNJ': 'Johnson & Johnson',
#     'RELIANCE.NS': 'Reliance Industries',
#     'TCS.NS': 'Tata Consultancy Services',
#     'INFY.NS': 'Infosys',
#     'HDFCBANK.NS': 'HDFC Bank',
#     'ICICIBANK.NS': 'ICICI Bank',
#     'SBIN.NS': 'State Bank of India',
#     'TITAN.NS': 'Titan Comapny Limited',
#     'SAIL.NS': 'Steel Authority of India Limited',
# }

# def validate_stock_symbol(symbol):
#     """Validate if stock symbol exists"""
#     try:
#         ticker = yf.Ticker(symbol)
#         hist = ticker.history(period='5d')
#         return not hist.empty
#     except:
#         return False

# def get_current_stock_price(symbol):
#     """Get current stock price"""
#     try:
#         ticker = yf.Ticker(symbol)
#         hist = ticker.history(period='1d')
#         if not hist.empty:
#             return hist['Close'].iloc[-1]
#     except:
#         pass
#     return None

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     result = None
    
#     if request.method == 'POST':
#         try:
#             # Get form data
#             symbol = request.form['symbol'].upper().strip()
#             lookback = int(request.form.get('lookback', 60))
#             epochs = int(request.form.get('epochs', 30))
#             period = request.form.get('period', '2y')
            
#             # Validate inputs
#             if not symbol:
#                 flash('Please enter a stock symbol', 'error')
#                 return render_template("index.html", popular_stocks=POPULAR_STOCKS)
            
#             if not (10 <= lookback <= 200):
#                 flash('Lookback days must be between 10 and 200', 'error')
#                 return render_template("index.html", popular_stocks=POPULAR_STOCKS)
            
#             if not (5 <= epochs <= 100):
#                 flash('Epochs must be between 5 and 100', 'error')
#                 return render_template("index.html", popular_stocks=POPULAR_STOCKS)
            
#             # Validate stock symbol
#             if not validate_stock_symbol(symbol):
#                 flash(f'Invalid stock symbol: {symbol}', 'error')
#                 return render_template("index.html", popular_stocks=POPULAR_STOCKS)
            
#             # Get current price for comparison
#             current_price = get_current_stock_price(symbol)
            
#             # Initialize and run prediction with web-optimized class
#             predictor = StockPredictorWeb(symbol=symbol, period=period)
            
#             # Suppress console output during training
#             old_stdout = sys.stdout
#             sys.stdout = buffer = io.StringIO()
            
#             try:
#                 # Run complete analysis
#                 analysis_result = predictor.run_complete_analysis_web(
#                     lookback_days=lookback, 
#                     epochs=epochs
#                 )
#             finally:
#                 sys.stdout = old_stdout
            
#             if not analysis_result:
#                 flash(f'Analysis failed for {symbol}', 'error')
#                 return render_template("index.html", popular_stocks=POPULAR_STOCKS)
            
#             # Prepare result data
#             result = {
#                 'symbol': symbol,
#                 'company_name': POPULAR_STOCKS.get(symbol, 'Unknown Company'),
#                 'current_price': current_price,
#                 'accuracy': f"{analysis_result['metrics']['accuracy']:.2f}",
#                 'rmse': f"{analysis_result['metrics']['rmse']:.2f}",
#                 'mae': f"{analysis_result['metrics']['mae']:.2f}",
#                 'predictions': [
#                     {
#                         'date': analysis_result['future_dates'][i].strftime('%Y-%m-%d'),
#                         'price': f"{analysis_result['future_predictions'][i][0]:.2f}"
#                     } for i in range(min(7, len(analysis_result['future_predictions'])))
#                 ],
#                 'data_range': f"{predictor.data.index[0].strftime('%Y-%m-%d')} to {predictor.data.index[-1].strftime('%Y-%m-%d')}",
#                 'training_params': {
#                     'lookback_days': lookback,
#                     'epochs': epochs,
#                     'period': period,
#                     'data_points': len(predictor.data)
#                 },
#                 'plots': analysis_result['plots']  # Add plots to result
#             }
            
#             # Calculate price change prediction
#             if current_price and len(analysis_result['future_predictions']) > 0:
#                 predicted_price = analysis_result['future_predictions'][0][0]
#                 price_change = predicted_price - current_price
#                 price_change_pct = (price_change / current_price) * 100
#                 result['price_change'] = f"{price_change:+.2f}"
#                 result['price_change_pct'] = f"{price_change_pct:+.2f}"
#                 result['trend'] = 'up' if price_change > 0 else 'down' if price_change < 0 else 'flat'
            
#             flash(f'Prediction completed for {symbol}!', 'success')
            
#         except Exception as e:
#             error_msg = str(e)
#             print(f"Error: {error_msg}")
#             print(traceback.format_exc())
#             flash(f'Error processing prediction: {error_msg}', 'error')
#             result = None
    
#     return render_template("index.html", result=result, popular_stocks=POPULAR_STOCKS)

# @app.route('/api/validate_symbol')
# def validate_symbol():
#     """API endpoint to validate stock symbol"""
#     symbol = request.args.get('symbol', '').upper().strip()
#     if not symbol:
#         return jsonify({'valid': False, 'message': 'No symbol provided'})
    
#     try:
#         ticker = yf.Ticker(symbol)
#         info = ticker.info
#         hist = ticker.history(period='5d')
        
#         if hist.empty:
#             return jsonify({'valid': False, 'message': 'No data available'})
        
#         return jsonify({
#             'valid': True,
#             'name': info.get('longName', 'Unknown'),
#             'current_price': hist['Close'].iloc[-1]
#         })
#     except Exception as e:
#         return jsonify({'valid': False, 'message': 'Invalid symbol'})

# @app.route('/about')
# def about():
#     """About page explaining the model"""
#     return render_template('about.html')

# if __name__ == '__main__':
#     # Create uploads directory if it doesn't exist
#     os.makedirs('static/plots', exist_ok=True)
#     app.run(debug=True, host='0.0.0.0', port=5000)

