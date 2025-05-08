import pandas as pd
import numpy as np
from prophet import Prophet
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up yfinance cache directory
import os
os.environ["YFINANCE_CACHE_DIR"] = "/tmp"

def fetch_data(tickers, start, end):
    try:
        data = yf.download(tickers, start=start, end=end)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
        return data
    except Exception as e:
        raise RuntimeError(f"Error fetching data: {str(e)}")

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(25, return_sequences=True, input_shape=input_shape),
        LSTM(25, return_sequences=False),
        Dense(10, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data(data, look_back=30):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def predict_stock_price_lstm(data, n_days):
    X, y, scaler = prepare_data(data)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    model = create_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, batch_size=32, epochs=30, validation_split=0.1, verbose=0)
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Inverse transform predictions
    train_predict = scaler.inverse_transform(train_predict)
    y_train = scaler.inverse_transform([y_train])
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform([y_test])
    
    # Calculate accuracy metrics
    train_rmse = np.sqrt(mean_squared_error(y_train[0], train_predict[:,0]))
    test_rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
    train_mae = mean_absolute_error(y_train[0], train_predict[:,0])
    test_mae = mean_absolute_error(y_test[0], test_predict[:,0])
    train_r2 = r2_score(y_train[0], train_predict[:,0])
    test_r2 = r2_score(y_test[0], test_predict[:,0])
    
    # Future predictions
    last_60_days = data[-60:]
    future_predictions = []
    
    for _ in range(n_days):
        X_future = scaler.transform(last_60_days.reshape(-1, 1))
        X_future = np.reshape(X_future, (1, X_future.shape[0], 1))
        prediction = model.predict(X_future)
        future_predictions.append(scaler.inverse_transform(prediction)[0][0])
        last_60_days = np.append(last_60_days[1:], prediction)
    
    return future_predictions, {
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Train MAE': train_mae,
        'Test MAE': test_mae,
        'Train R2': train_r2,
        'Test R2': test_r2
    }

def predict_stock_price_prophet(data, n_days):
    df = pd.DataFrame({'ds': data.index, 'y': data.values})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=n_days)
    forecast = model.predict(future)
    
    # Calculate accuracy metrics
    train_predictions = forecast['yhat'][:len(df)]
    train_rmse = np.sqrt(mean_squared_error(df['y'], train_predictions))
    train_mae = mean_absolute_error(df['y'], train_predictions)
    train_r2 = r2_score(df['y'], train_predictions)
    
    return forecast['yhat'].tail(n_days).values, {
        'Train RMSE': train_rmse,
        'Train MAE': train_mae,
        'Train R2': train_r2
    }
