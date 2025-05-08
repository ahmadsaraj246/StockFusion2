import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import time

from stockfusion_logic import fetch_data, predict_stock_price_lstm, predict_stock_price_prophet

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

def add_meta_tag():
    meta_tag = """
    <head>
    <meta name="google-site-verification" content="QBiAoAo1GAkCBe1QoWq-dQ1RjtPHeFPyzkqJqsrqW-s" />
    </head>
    """
    st.markdown(meta_tag, unsafe_allow_html=True)

st.write('''# StockFusion ''')
st.sidebar.write('''# StockFusion ''')

with st.sidebar:
    selected = option_menu("Utilities", ["Stocks Performance Comparison", "Real-Time Stock Price", "Stock Prediction", "About Developers"])
    start = st.sidebar.date_input('Start', datetime.date(2015, 1, 1))
    end = st.sidebar.date_input('End', datetime.date.today())

if selected == 'Stocks Performance Comparison':
    st.subheader("Stocks Performance Comparison")
    ticker_input = st.text_input("Enter ticker symbols (comma-separated, e.g., AAPL,MSFT,GOOGL)")
    if ticker_input:
        tickers = [ticker.strip().upper() for ticker in ticker_input.split(',')]
        with st.spinner('Loading...'):
            data = fetch_data(tickers, start, end)
            data = data.sort_index(ascending=False)  # Sort by index in descending order
            if not data.empty:
                st.subheader('Data')
                st.write(data)

                # Combined Close Price Graph
                st.subheader('Close Price')
                fig_close = go.Figure()
                for ticker in tickers:
                    fig_close.add_trace(go.Scatter(x=data.index, y=data[f'Close_{ticker}'], name=f'{ticker} Close'))
                st.plotly_chart(fig_close)

                # Combined Open Price Graph
                st.subheader('Open Price')
                fig_open = go.Figure()
                for ticker in tickers:
                    fig_open.add_trace(go.Scatter(x=data.index, y=data[f'Open_{ticker}'], name=f'{ticker} Open'))
                st.plotly_chart(fig_open)

                # Combined Volume Graph
                st.subheader('Volume')
                fig_volume = go.Figure()
                for ticker in tickers:
                    fig_volume.add_trace(go.Bar(x=data.index, y=data[f'Volume_{ticker}'], name=f'{ticker} Volume'))
                st.plotly_chart(fig_volume)
            else:
                st.error("No data found for these ticker symbols")
    else:
        st.write('Please enter at least one ticker symbol')

elif selected == 'Real-Time Stock Price':
    st.subheader("Real-Time Stock Price")
    ticker = st.text_input("Enter ticker symbol (e.g., AAPL)")
    if st.button("Search") and ticker:
        with st.spinner('Loading...'):
            data = fetch_data([ticker], start, end)
            data = data.sort_index(ascending=False)  # Sort by index in descending order
            if not data.empty:
                data.reset_index(inplace=True)
                st.subheader(f'Data of {ticker}')
                st.write(data)

                def plot_raw_data():
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data['Date'], y=data[f'Open_{ticker}'], name="stock_open"))
                    fig.add_trace(go.Scatter(x=data['Date'], y=data[f'Close_{ticker}'], name="stock_close"))
                    fig.layout.update(title_text=f' Line Chart of {ticker}', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig)

                def plot_candle_data():
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=data['Date'], open=data[f'Open_{ticker}'], high=data[f'High_{ticker}'], low=data[f'Low_{ticker}'], close=data[f'Close_{ticker}'], name='market data'))
                    fig.update_layout(title=f'Candlestick Chart of {ticker}', yaxis_title='Stock Price', xaxis_title='Date')
                    st.plotly_chart(fig)

                chart = ('Candle Stick')
                dropdown1 = st.selectbox('Pick your chart', chart)
                if dropdown1 == 'Candle Stick':
                    plot_candle_data()
            else:
                st.error("No data found for this ticker symbol")

elif selected == 'Stock Prediction':
    st.subheader("Stock Prediction")
    ticker = st.text_input("Enter ticker symbol (e.g., AAPL)")
    
    if ticker:
        with st.spinner('Loading...'):
            data = fetch_data([ticker], start, end)
            
            if not data.empty:
                data.reset_index(inplace=True)
                st.subheader(f'Data of {ticker}')
                st.write(data)

                def plot_raw_data():
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data['Date'], y=data[f'Open_{ticker}'], name="Stock Open"))
                    fig.add_trace(go.Scatter(x=data['Date'], y=data[f'Close_{ticker}'], name="Stock Close"))
                    fig.layout.update(title_text=f'Time Series Data of {ticker}', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig)

                plot_raw_data()
                
                n_days = st.slider('Days of prediction:', 1, 365)
                
                model_choice = st.radio("Choose prediction model:", ("Prophet", "LSTM"))

                close_prices = data[f'Close_{ticker}'].values
                if model_choice == 'LSTM':
                    with st.spinner("Training LSTM model..."):
                        predictions, metrics = predict_stock_price_lstm(close_prices, n_days)
                    
                    st.subheader("LSTM Prediction Results")
                    st.write(f"Predicted prices for the next {n_days} days:")
                                    
                    # Create a DataFrame with actual and predicted prices
                    prediction_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_days)
                    pred_df = pd.DataFrame({
                        'Date': prediction_dates,
                        'Predicted Close': predictions,
                        'Actual Close': [None] * n_days  # We don't have actual values for future dates
                    })
                    # Display the prediction table
                    st.write(pred_df)
                    
                    st.subheader("Model Performance Metrics")
                    for metric, value in metrics.items():
                        st.write(f"{metric}: {value:.4f}")
                    # Calculate overall accuracy
                    accuracy = (metrics['Train R2'] + metrics['Test R2']) / 2 * 100
                    st.write(f"Overall Accuracy: {accuracy:.2f}%")
                    
                    # Plot actual vs predicted
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data['Date'], y=data[f'Close_{ticker}'], name='Actual Close Price', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=pd.date_range(start=data['Date'].iloc[-1], periods=n_days+1, freq='D')[1:], 
                                             y=predictions, name='Predicted Close Price', line=dict(color='red', dash='dot')))
                    fig.update_layout(title=f'{ticker} Stock Price Prediction', xaxis_title='Date', yaxis_title='Close Price')
                    st.plotly_chart(fig)

                elif model_choice == 'Prophet':
                    with st.spinner("Training Prophet model..."):
                        predictions, metrics = predict_stock_price_prophet(pd.Series(close_prices, index=data['Date']), n_days)
                    
                    st.subheader("Prophet Prediction Results")
                    st.write(f"Predicted prices for the next {n_days} days:")
                     # Create a DataFrame with actual and predicted prices
                    prediction_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_days)
                    pred_df = pd.DataFrame({
                        'Date': prediction_dates,
                        'Predicted Close': predictions,
                        'Actual Close': [None] * n_days  # We don't have actual values for future dates
                    })
                    # Display the prediction table
                    st.write(pred_df)

                    
                    st.subheader("Model Performance Metrics")
                    for metric, value in metrics.items():
                        st.write(f"{metric}: {value:.4f}")
                     # Calculate overall accuracy
                    accuracy = metrics['Train R2'] * 100
                    st.write(f"Overall Accuracy: {accuracy:.2f}%")
                    
                    
                    # Plot actual vs predicted
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=data['Date'], y=data[f'Close_{ticker}'], name='Actual Price', line=dict(color='blue')))
                    fig.add_trace(go.Scatter(x=pd.date_range(start=data['Date'].iloc[-1], periods=n_days+1, freq='D')[1:], 
                                             y=predictions, name='Predicted Price', line=dict(color='red', dash='dot')))
                    fig.update_layout(title=f'{ticker} Stock Price Prediction', xaxis_title='Date', yaxis_title='Close Price', )
                    st.plotly_chart(fig)

            else:
                st.error("No data found for this ticker symbol")
                    
if selected == 'About Developers':
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: #1E88E5;'>About the Developers</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    st.markdown("---")
    with col1:
        st.image("https://img.icons8.com/color/240/000000/user-male-circle--v1.png", width=150)
    with col2:
        st.markdown("<h3 style='color: #1E88E5;'>Ahmad Saraj and Arbaz Shafiq</h3>", unsafe_allow_html=True)
        st.markdown("<p style='font-style: italic;'>Financial Data Analyst & Machine Learning Enthusiast</p>", unsafe_allow_html=True)
        st.markdown("Passionate about leveraging data science and machine learning to unlock insights in financial markets.")

    st.markdown("<h4 style='text-align: center; color: #1E88E5; margin-top: 20px;'>Connect with Us</h4>", unsafe_allow_html=True)

    social_media = {
        "Ahmad Saraj": {
            "LinkedIn": {"url": "https://www.linkedin.com/in/ahmad-saraj-69a73a292/", "icon": "https://img.icons8.com/color/48/000000/linkedin.png"},
            "GitHub": {"url": "https://github.com/ahmadsaraj246", "icon": "https://img.icons8.com/fluent/48/000000/github.png"}
        },
        "Arbaz Shafiq": {
            "LinkedIn": {"url": "https://www.linkedin.com/in/arbaz-shafiq-802374267/", "icon": "https://img.icons8.com/color/48/000000/linkedin.png"},
            "GitHub": {"url": "https://github.com/malik087", "icon": "https://img.icons8.com/fluent/48/000000/github.png"}
        }
    }

    for developer, links in social_media.items():
        st.markdown(f"<h4 style='text-align: center; color: #1E88E5; margin-top: 20px;'>{developer}</h4>", unsafe_allow_html=True)
        cols = st.columns(len(links))
        for index, (platform, info) in enumerate(links.items()):
            with cols[index]:
                st.markdown(f"""
                <a href="{info['url']}" target="_blank">
                    <img src="{info['icon']}" width="40" height="40" style="display: block; margin: auto;">
                    <p style="text-align: center; font-size: 0.8em; margin-top: 5px;">{platform}</p>
                </a>
                """, unsafe_allow_html=True)

    st.markdown("<p style='text-align: center; font-size: 0.8em; margin-top: 30px;'>© 2025 Ahmad Saraj. All rights reserved.</p>", unsafe_allow_html=True)
