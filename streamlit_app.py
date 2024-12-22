import streamlit as st
import requests
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta

# API settings
API_URL = "https://www.alphavantage.co/query"
API_KEY = "RFIRS1QW21OFFHVZ"

def get_stock_data(symbol):
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": API_KEY,
        "outputsize": "compact"
    }
    response = requests.get(API_URL, params=params)
    data = response.json()
    if "Time Series (Daily)" not in data:
        st.error("Couldn't get the stock price data. Please check the stock symbol that you entered")
        return None
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"], orient="index")
    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close",
        "5. volume": "volume"
    })
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df

def train_and_predict(data, days_ahead=7):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[["close"]])

    # Create sequences for LSTM
    sequence_length = 60
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length - days_ahead):
        X.append(data_scaled[i:i+sequence_length, 0])
        y.append(data_scaled[i+sequence_length+days_ahead, 0])
    X, y = np.array(X), np.array(y)

    # Reshape X for LSTM input
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Train-test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    # Predict the next price
    last_sequence = data_scaled[-sequence_length:].reshape(1, sequence_length, 1)
    scaled_prediction = model.predict(last_sequence)[0, 0]
    prediction = scaler.inverse_transform([[scaled_prediction]])[0, 0]
    return prediction

# Streamlit interface
st.title("Stock Price Prediction Chatbot")
symbol = st.text_input("Please write a stock symbol... (e.g., AAPL):", value="AAPL")
days_ahead = st.slider("Prediction period (days):", min_value=1, max_value=30, value=7)

if st.button("Predict"):
    with st.spinner("Stock data is processing..."):
        data = get_stock_data(symbol)
        if data is not None:
            st.line_chart(data["close"])
            prediction = train_and_predict(data, days_ahead=days_ahead)
            st.success(f"For symbol {symbol}, prediction after {days_ahead} days: ${prediction:.2f}")
