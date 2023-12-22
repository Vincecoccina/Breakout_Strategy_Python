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

client = Client(api_key=api_key, api_secret=secret_key, tld='com', testnet=False)

def get_recent(symbol, interval):
        
        start = "2023-08-01"

        data = pd.DataFrame(client.get_historical_klines(symbol=symbol, interval=interval, start_str=start))
        data["Date"] = pd.to_datetime(data.iloc[:,0], unit = "ms")
        data.columns = ["Open Time", "Open", "High", "Low", "Close", "Volume",
                      "Clos Time", "Quote Asset Volume", "Number of Trades",
                      "Taker Buy Base Asset Volume", "Taker Buy Quote Asset Volume", "Ignore", "Date"]
        data = data[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        data.set_index("Date", inplace = True)
        data = data.astype(float)
        return data

df = get_recent("BTCUSDT", "1h")
df.reset_index(drop=True, inplace=True)

def calculate_ema(data, length: int):
        return ta.ema(data["Close"], length)
        
df['EMA_Short'] = calculate_ema(df, 20)
df['EMA_Long'] = calculate_ema(df, 50)
df.dropna(inplace=True)
print(df)
