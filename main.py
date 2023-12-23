import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
import os
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

load_dotenv()
api_key = os.getenv("API_KEY_TEST")
secret_key = os.getenv("SECRET_KEY_TEST")

# Binance Client
client = Client(api_key=api_key, api_secret=secret_key, tld='com', testnet=False)

# Get token's historical datas
def get_recent(symbol, interval, start):
        data = pd.DataFrame(client.get_historical_klines(symbol=symbol, interval=interval, start_str=start))
        data["Date"] = pd.to_datetime(data.iloc[:,0], unit = "ms")
        data.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                      "Clos Time", "Quote Asset Volume", "Number of Trades",
                      "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
        data = data[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        data.set_index("Date", inplace = True)
        data = data.astype(float)
        return data

# Calculate EMA
def calculate_ema(data, length: int):
        return ta.ema(data["Close"], length)

# Calculate MACD (confirm Trend)
def calculate_macd(data, fast_length, slow_length, signal_length):
    macd = ta.macd(data["Close"], fast=fast_length, slow=slow_length, signal=signal_length)
    return macd['MACD_12_26_9'], macd['MACDh_12_26_9'], macd['MACDs_12_26_9']

# Generate CROSSOVER signal
def generate_crossover_signal(data):
     data["EMA_Trend"] = 0.0
     data["EMA_Trend"] = np.where(data["EMA_Short"] > data["EMA_Long"], 1.0, 0.0)
     data["Crossover_Trend"] = data["EMA_Trend"].diff()

     return data

# Generate MACD signal
def generate_macd_signal(data):
    macd_bullish = (data['MACD'] > data['MACD_Signal'])
    macd_bearish = (data['MACD'] < data['MACD_Signal'])

    data["MACD_Trend"] = np.where(macd_bullish, 1, 
                         np.where(macd_bearish, -1, 0))

    return data

# Combine CROSSOVER signal with MACD signal for generate a entry signal
def generate_entry_signal(data):
    buy_signal = (data["Crossover_Trend"] > 0) & (data["MACD_Trend"] == 1) # Buy signal
    sell_signal = (data["Crossover_Trend"] < 0) & (data["MACD_Trend"] == -1) # Sell signal

    data["Entry"] = np.where(buy_signal, 1, 
                    np.where(sell_signal, -1, 0))

    return data




df = get_recent("BTCUSDT", "1h", "2023-01-01")
# --------------------------------------------
# Calculate indicators    
df['EMA_Short'] = calculate_ema(df, 20)
df['EMA_Long'] = calculate_ema(df, 50)
df['MACD'], df['MACD_Histogram'], df['MACD_Signal'] = calculate_macd(df, 12, 26, 9)

df = generate_crossover_signal(df)
df = generate_macd_signal(df)
df = generate_entry_signal(df)
df.dropna(inplace=True)

df_filtered = df[df["Entry"]!=0]
print(df_filtered)


# Créer le graphique à chandeliers
fig = go.Figure(data=[go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"])])
        # Ajouter des lignes pour les indicateurs
fig.add_trace(go.Scatter(x=df.index, y=df["EMA_Short"], mode="lines", name="EMA_Short", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=df.index, y=df["EMA_Long"], mode="lines", name="EMA_Long", line=dict(color="crimson")))
fig.add_trace(go.Scatter(x=df[df["Entry"] == 1].index, y=df[df["Entry"] == 1]["Close"], 
                         mode="markers", marker_symbol="triangle-up", marker_color="green",marker_size=15, name="Buy"))
fig.add_trace(go.Scatter(x=df[df["Entry"] == -1].index, y=df[df["Entry"] == -1]["Close"], 
                         mode="markers", marker_symbol="triangle-down", marker_color="red",marker_size=15, name="Sell"))

      
# Afficher le graphique
fig.show()
