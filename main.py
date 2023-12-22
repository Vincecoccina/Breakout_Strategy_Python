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
df['EMA'] = ta.ema(df.Close, length=50)
df.dropna(subset=['EMA'], inplace=True)

EMAsignal = [0]*len(df)
backcandles = 10

# Votre code précédent reste inchangé jusqu'à la boucle for
for row in range(backcandles, len(df)):
    upt = 1
    dnt = 1
    for i in range(row-backcandles, row+1):
        if max(df.iloc[i]['Open'], df.iloc[i]['Close']) >= df.iloc[i]['EMA']:
            dnt = 0
        if min(df.iloc[i]['Open'], df.iloc[i]['Close']) <= df.iloc[i]['EMA']:
            upt = 0
    if upt == 1 and dnt == 1:
        EMAsignal[row] = 3
    elif upt == 1:
        EMAsignal[row] = 2
    elif dnt == 1:
        EMAsignal[row] = 1

df['EMASignal'] = EMAsignal

def isPivot(candle, window):
    """
    function that detects if a candle is a pivot/fractal point
    args: candle index, window before and after candle to test if pivot
    returns: 1 if pivot high, 2 if pivot low, 3 if both and 0 default
    """
    if candle-window < 0 or candle+window >= len(df):
        return 0
    
    pivotHigh = 1
    pivotLow = 2
    for i in range(candle-window, candle+window+1):
        if df.iloc[candle].Low > df.iloc[i].Low:
            pivotLow=0
        if df.iloc[candle].High < df.iloc[i].High:
            pivotHigh=0
    if (pivotHigh and pivotLow):
        return 3
    elif pivotHigh:
        return pivotHigh
    elif pivotLow:
        return pivotLow
    else:
        return 0

window=6
df['isPivot'] = df.apply(lambda x: isPivot(x.name,window), axis=1)
df = df[df["isPivot"]!=0]
df = df[df["EMASignal"]!=0]

   

print(df)
