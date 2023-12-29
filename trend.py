import pandas as pd
import pandas_ta as ta
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
import os
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance import ThreadedWebsocketManager
from backtesting import Backtest, Strategy
from dotenv import load_dotenv
import time

load_dotenv()

api_key = os.getenv("API_KEY_TEST")
secret_key = os.getenv("SECRET_KEY_TEST")

# Binance Client
client = Client(api_key=api_key, api_secret=secret_key, tld='com', testnet=False)

def get_historical(symbol, interval, start):

        df = pd.DataFrame(client.get_historical_klines(symbol=symbol, interval=interval, start_str=start))
        df["Date"] = pd.to_datetime(df.iloc[:,0], unit = "ms")
        df.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                      "Clos Time", "Quote Asset Volume", "Number of Trades",
                      "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        df.set_index("Date", inplace = True)
        df[['High', 'Low', 'Close', 'Volume']] = df[['High', 'Low', 'Close', 'Volume']].astype(float)
        for column in df.columns:
                df[column] = pd.to_numeric(df[column], errors = "coerce")
        df["Complete"] = [True for row in range(len(df)-1)] + [False]

        return df

def calculate_sma(data, length: int):
    return ta.sma(data['Close'], length)

# Identify Trend Direction
def calculate_slope(series, period: int = 5):
    slopes = [0 for _ in range(period-1)]
    for i in range(period-1, len(series)):
        x = np.arange(period)
        y = series[i-period+1:i+1].values
        slope = np.polyfit(x, y, 1)[0]
        percent_slope = (slope / y[0]) * 100
        slopes.append(percent_slope)
    return slopes

# Identify Trend
def determinate_trend(data):
    if data["SMA_10"] > data["SMA_20"] > data["SMA_30"]:
        return 2 # Tendance haussière
    elif data["SMA_10"] < data["SMA_20"] < data["SMA_30"]:
        return 1 # Tendance baissière
    else:
        return 0 # Pas de tendance

# Fonction qui compare la bougie à la moyenne mobile
def check_candles(data, backcandles, ma_column):
    categories = [0 for _ in range(backcandles)]
    for i in range(backcandles, len(data)):
        if all(data["Close"][i-backcandles:i] > data[ma_column][i-backcandles:i]):
            categories.append(2) # Tendance Haussière
        elif all(data["Close"][i-backcandles:i] < data[ma_column][i-backcandles:i]):
            categories.append(1) #Tendance Baissière
        else:
            categories.append(0)

    return categories

# Fonction qui génère un signal pour la tendnace basé sur l'ADX
def generate_trend_signal(data, threshold=40):
    trend_signal = []
    for i in range(len(data)):
        if data["ADX"][i] > threshold:
            if data["DMP"][i] > data["DMN"][i]:
                trend_signal.append(2) # Tendance haussière
            else:
                trend_signal.append(1) # Tendance Baissière
        else:
            trend_signal.append(0) # Pas de tendance claire
    return trend_signal



data = get_historical("NEARUSDT", "1d", "2023-09-01")
data["SMA_10"] = calculate_sma(data, 10)
data["SMA_20"] = calculate_sma(data, 20)
data["SMA_30"] = calculate_sma(data, 30)
data["Slope"] = calculate_slope(data["SMA_20"])
data["Trend"] = data.apply(determinate_trend, axis=1)
data["Category"] = check_candles(data, 5, "SMA_20")
data.ta.adx(append=True)
data = data.rename(columns=lambda x: x[:3] if x.startswith("ADX") else x)
data = data.rename(columns=lambda x: x[:3] if x.startswith("DM") else x)
data["Trend Signal"] = generate_trend_signal(data)
data["Confirmed Signal"] = data.apply(lambda row: row["Category"] if row["Category"] == row["Trend Signal"] else 0, axis=1)
data.dropna(inplace=True)


data = data[data["Confirmed Signal"]!=0]
print(data)