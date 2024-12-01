import streamlit as st
import requests
import pandas as pd
import numpy as np
import sklearn as sl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# API settings
API_URL = "https://www.alphavantage.co/query"
API_KEY = "YOUR_API_KEY"

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
    # Price scoring for machine learning
    data["target"] = data["close"].shift(-days_ahead)
    data = data.dropna()
    X = data[["open", "high", "low", "close", "volume"]]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Future prediction
    future_data = X.iloc[-1:].values  # Son veriyi kullanarak tahmin yap
    prediction = model.predict(future_data)[0]
    return prediction

# Streamlit interface
st.title("Stock Price Prediction Chatbot")
symbol = st.text_input("Please write a stock symbol... (egg. AAPL):", value="AAPL")
days_ahead = st.slider("Tahmin süresi (gün):", min_value=1, max_value=30, value=7)

if st.button("Predict"):
    with st.spinner("Stock data is processing..."):
        data = get_stock_data(symbol)
        if data is not None:
            st.line_chart(data["close"])
            prediction = train_and_predict(data, days_ahead=days_ahead)
            st.success(f"for symbol {symbol} prediction after {days_ahead} days: ${prediction:.2f}")
