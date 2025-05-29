import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import base64
import io
from datetime import datetime, timedelta
import sys
import traceback

# Import your existing predictor class
from spp2 import StockPredictorWeb

# Configure Streamlit page
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Popular stocks dictionary
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

@st.cache_data
def validate_stock_symbol(symbol):
    """Validate if stock symbol exists"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='5d')
        return not hist.empty
    except:
        return False

@st.cache_data
def get_current_stock_price(symbol):
    """Get current stock price"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='1d')
        if not hist.empty:
            return hist['Close'].iloc[-1]
    except:
        pass
    return None

def display_base64_image(base64_string, width=None):
    """Display base64 encoded image in Streamlit"""
    if base64_string:
        image_html = f'<img src="data:image/png;base64,{base64_string}" style="width:100%; max-width:800px;">'
        st.markdown(image_html, unsafe_allow_html=True)

def main():
    st.title("🤖 AI Stock Predictor")
    st.markdown("### Predict stock prices using LSTM Neural Networks")
    
    # Sidebar for inputs
    st.sidebar.header("📊 Configuration")
    
    # Stock selection
    st.sidebar.subheader("Stock Selection")
    
    # Popular stocks dropdown
    popular_choice = st.sidebar.selectbox(
        "Choose a popular stock:",
        [""] + list(POPULAR_STOCKS.keys()),
        format_func=lambda x: f"{x} - {POPULAR_STOCKS.get(x, '')}" if x else "Select a stock..."
    )
    
    # Manual stock input
    manual_symbol = st.sidebar.text_input(
        "Or enter stock symbol manually:",
        placeholder="e.g., AAPL, GOOGL"
    ).upper().strip()
    
    # Determine which symbol to use
    symbol = manual_symbol if manual_symbol else popular_choice
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    lookback = st.sidebar.slider("Lookback Days", 10, 200, 60, 
                                help="Number of previous days to use for prediction")
    epochs = st.sidebar.slider("Training Epochs", 5, 100, 30,
                              help="Number of training iterations")
    period = st.sidebar.selectbox("Data Period", 
                                 ["1y", "2y", "5y", "max"],
                                 index=1,
                                 help="Historical data period to fetch")
    
    # Prediction button
    predict_button = st.sidebar.button("🚀 Run Prediction", type="primary")
    
    # Main content area
    if not symbol:
        st.info("👆 Please select a stock symbol from the sidebar to get started!")
        
        # Display some info about the app
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 How it works")
            st.write("""
            1. **Select a stock** from popular choices or enter manually
            2. **Configure parameters** like lookback days and epochs
            3. **Run prediction** to train an LSTM neural network
            4. **View results** including accuracy metrics and future predictions
            """)
        
        with col2:
            st.subheader("🎯 Features")
            st.write("""
            - **LSTM Neural Network** for time series prediction
            - **Real-time data** from Yahoo Finance
            - **Interactive visualizations** of predictions
            - **Performance metrics** and accuracy scores
            - **Future price forecasts** up to 30 days
            """)
        
        return
    
    # Validate symbol before prediction
    if symbol and not predict_button:
        with st.spinner(f"Validating {symbol}..."):
            if validate_stock_symbol(symbol):
                current_price = get_current_stock_price(symbol)
                company_name = POPULAR_STOCKS.get(symbol, "Unknown Company")
                
                st.success(f"✅ {symbol} - {company_name}")
                if current_price:
                    st.metric("Current Price", f"${current_price:.2f}")
            else:
                st.error(f"❌ Invalid stock symbol: {symbol}")
                return
    
    # Run prediction
    if predict_button and symbol:
        if not validate_stock_symbol(symbol):
            st.error(f"❌ Invalid stock symbol: {symbol}")
            return
        
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔄 Initializing predictor...")
            progress_bar.progress(10)
            
            # Initialize predictor
            predictor = StockPredictorWeb(symbol=symbol, period=period)
            
            status_text.text("📊 Fetching stock data...")
            progress_bar.progress(20)
            
            # Suppress console output
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            try:
                status_text.text("🧠 Training neural network...")
                progress_bar.progress(40)
                
                # Run analysis
                analysis_result = predictor.run_complete_analysis_web(
                    lookback_days=lookback, 
                    epochs=epochs
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
            finally:
                sys.stdout = old_stdout
            
            if not analysis_result:
                st.error(f"❌ Analysis failed for {symbol}")
                return
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.success(f"🎉 Prediction completed for {symbol}!")
            
            # Get current price for comparison
            current_price = get_current_stock_price(symbol)
            company_name = POPULAR_STOCKS.get(symbol, "Unknown Company")
            
            # Main metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Company", company_name)
            
            with col2:
                if current_price:
                    st.metric("Current Price", f"${current_price:.2f}")
            
            with col3:
                st.metric("Accuracy", f"{analysis_result['metrics']['accuracy']:.1f}%",
                         help="Percentage of predictions within 5% of actual price")
            
            with col4:
                st.metric("RMSE", f"{analysis_result['metrics']['rmse']:.2f}",
                         help="Root Mean Square Error")
            
            # Future predictions
            st.subheader("🔮 Future Price Predictions")
            
            predictions_data = []
            for i in range(min(7, len(analysis_result['future_predictions']))):
                date = analysis_result['future_dates'][i].strftime('%Y-%m-%d')
                price = analysis_result['future_predictions'][i][0]
                predictions_data.append({
                    'Date': date,
                    'Predicted Price': f"${price:.2f}",
                    'Day': f"Day +{i+1}"
                })
            
            predictions_df = pd.DataFrame(predictions_data)
            st.dataframe(predictions_df, use_container_width=True, hide_index=True)
            
            # Price change analysis
            if current_price and len(analysis_result['future_predictions']) > 0:
                predicted_price = analysis_result['future_predictions'][0][0]
                price_change = predicted_price - current_price
                price_change_pct = (price_change / current_price) * 100
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tomorrow's Prediction", f"${predicted_price:.2f}")
                with col2:
                    st.metric("Price Change", f"{price_change:+.2f}", 
                             delta=f"{price_change:+.2f}")
                with col3:
                    st.metric("Change %", f"{price_change_pct:+.2f}%",
                             delta=f"{price_change_pct:+.2f}%")
            
            # Visualizations
            st.subheader("📊 Analysis Visualizations")
            
            # Create tabs for different plots
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Price History", "🧠 Training", "🎯 Predictions", "🔮 Future"])
            
            with tab1:
                st.write("Historical stock price and trading volume")
                display_base64_image(analysis_result['plots']['price_history'])
            
            with tab2:
                st.write("Neural network training progress")
                display_base64_image(analysis_result['plots']['training'])
            
            with tab3:
                st.write("Model predictions vs actual prices")
                display_base64_image(analysis_result['plots']['predictions'])
            
            with tab4:
                st.write("Future price predictions")
                display_base64_image(analysis_result['plots']['future'])
            
            # Technical details
            with st.expander("🔧 Technical Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Model Parameters:**")
                    st.write(f"- Lookback Days: {lookback}")
                    st.write(f"- Training Epochs: {epochs}")
                    st.write(f"- Data Period: {period}")
                    st.write(f"- Data Points: {len(predictor.data)}")
                
                with col2:
                    st.write("**Performance Metrics:**")
                    st.write(f"- Accuracy: {analysis_result['metrics']['accuracy']:.2f}%")
                    st.write(f"- RMSE: {analysis_result['metrics']['rmse']:.4f}")
                    st.write(f"- MAE: {analysis_result['metrics']['mae']:.4f}")
                    st.write(f"- Data Range: {predictor.data.index[0].strftime('%Y-%m-%d')} to {predictor.data.index[-1].strftime('%Y-%m-%d')}")
            
        except Exception as e:
            st.error(f"❌ Error processing prediction: {str(e)}")
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()