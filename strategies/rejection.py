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

# -------------------------------------------
# Fonction qui identifie la bougie de rejet (identifie un possible retournement du marché)
def identify_rejection(data):
    data["Rejection"] = data.apply(lambda row: 2 if(
        ((min(row["Open"], row["Close"]) - row["Low"]) > (1.5 * abs(row["Close"] - row["Open"]))) and
        (row["High"] - max(row["Close"], row["Open"])) < (0.8 * abs(row["Close"] - row["Open"])) and
        (abs(row["Open"] - row["Close"]) > row["Open"] * 0.001)
    )else 1 if(
        (row["High"] - max(row["Open"], row["Close"])) > (1.5 * abs(row["Open"] - row["Close"])) and
        (min(row["Close"], row["Open"]) - row["Low"]) < (0.8 * abs(row["Open"] - row["Close"])) and
        (abs(row["Open"] - row["Close"]) > row["Open"] * 0.001)
    ) else 0, axis=1)

    return data

# -----------------------------------------------------------
# Fonctions qui definissent le support et la resistance

def support(df1, l, n1, n2):
    if df1.iloc[l-n1:l]["Low"].min() < df1.iloc[l]["Low"] or df1.iloc[l+1:l+n2+1]["Low"].min() < df1.iloc[l]["Low"]:
        return 0
    return 1
'''
La fonction support détermine si une ligne spécifique dans le DataFrame représente un niveau de support.
    df1: Le DataFrame contenant les données de marché.
    l: L'index actuel de la ligne à évaluer.
    n1: Le nombre de périodes avant la période actuelle à considérer.
    n2: Le nombre de périodes après la période actuelle à considérer.

    La fonction vérifie si le prix le plus bas (Low) de la période actuelle (l) est supérieur au prix le plus bas des n1 périodes précédentes et des n2 périodes suivantes. Si c'est le cas, cela suggère que le prix actuel est un support, puisqu'il n'y a pas eu de prix plus bas récemment avant ou juste après, et la fonction retourne 1. Si un prix plus bas est trouvé soit avant soit après la période actuelle, la fonction retourne 0, indiquant que ce n'est pas un niveau de support.
'''


def resistance(df1, l, n1, n2):
    if df1.iloc[l-n1:l]["High"].max() > df1.iloc[l]["High"] or df1.iloc[l+1:l+n2+1]["High"].max() > df1.iloc[l]["High"]:
        return 0
    return 1

'''
La fonction resistance fonctionne de manière similaire mais pour identifier les niveaux de résistance.
    df1: Le DataFrame contenant les données de marché.
    l: L'index actuel de la ligne à évaluer.
    n1: Le nombre de périodes avant la période actuelle à considérer.
    n2: Le nombre de périodes après la période actuelle à considérer.

    La fonction vérifie si le prix le plus haut (High) de la période actuelle (l) est inférieur au prix le plus haut des n1 périodes précédentes et n'est pas dépassé par les prix des n2 périodes suivantes. Si c'est le cas, cela suggère que le prix actuel est une résistance. Si le prix le plus haut de la période actuelle est dépassé par les prix avant ou après, la fonction retourne 0, indiquant que ce n'est pas un niveau de résistance.
'''

# -----------------------------------------------------------
# Fonctions qui detecte si un prix est proche d'un niveau de resistance ou de support
def close_resistance(l, levels, lim, df):
    if len(levels) == 0:
        return 0
    high = df.iloc[l]["High"]
    open_price = df.iloc[l]["Open"]
    close_price = df.iloc[l]["Close"]
    low = df.iloc[l]["Low"]
    closest_level = min(levels, key=lambda x: abs(x - high))

    c1 = abs(high - closest_level) <= lim
    c2 = abs(max(open_price, close_price) - closest_level) <= lim
    c3 = min(open_price, close_price) < closest_level
    c4 = low < closest_level

    if (c1 or c2) and c3 and c4:
        return closest_level
    else:
        return 0
    
def close_support(l, levels, lim, df):
    if len(levels) == 0:
        return 0
    low = df.iloc[l]["Low"]
    open_price = df.iloc[l]["Open"]
    close_price = df.iloc[l]["Close"]
    high = df.iloc[l]["High"]
    closest_level = min(levels, key=lambda x: abs(x - low))

    c1 = abs(low - closest_level) <= lim
    c2 = abs(max(open_price, close_price) - closest_level) <= lim
    c3 = min(open_price, close_price) < closest_level
    c4 = high < closest_level

    if (c1 or c2) and c3 and c4:
        return closest_level
    else:
        return 0

# Fonctions qui detecte si le prix est en-dessous de la resistance et si le prix est au-dessus du support
def is_below_resistance(l, level_backCandles, level, df):
    return df.iloc[l-level_backCandles:l]["High"].max() < level

def is_above_support(l, level_backCandles, level, df):
    return df.iloc[l-level_backCandles:l]["Low"].min() > level


# -----------------------------------------------------------
# Fonctions qui génèrent un signal d'entrée
def check_candle_signal(l, n1, n2, levelBackCandles, windowsBackCandles, df):
    ss = []
    rr = []
    for subrow in range(l-levelBackCandles, l-n2+1):
        if support(df, subrow, n1, n2):
            ss.append(df.iloc[subrow]["Low"])
        if resistance(df, subrow, n1, n2):
            rr.append(df.iloc[subrow]["High"])
    
    ss.sort()
    for i in range(1, len(ss)):
        if(i>=len(ss)):
            break
        if abs(ss[i] - ss[i-1])/ss[i]<=0.001:
            ss.pop(i)
    
    rr.sort()
    for i in range(1, len(rr)):
        if(i>=len(rr)):
            break
        if abs(rr[i] - rr[i-1])/rr[i]<=0.001:
            rr.pop(i)
    
    #-----------------------------------------------
    rrss = rr+ss
    rrss.sort()
    for i in range(1, len(rrss)):
        if(i>=len(rrss)):
            break
        if abs(rrss[i]-rrss[i-1])/rrss[i]<=0.001:
            rrss.pop(i)
    close_price = df.iloc[l]["Close"]
    cR = close_resistance(l, rrss, close_price * 0.003, df)
    cS = close_support(l, rrss, close_price * 0.003, df)

    #-------------------------------------------------
    if df.iloc[l]["Rejection"] == 1 and cR and is_below_resistance(l, windowsBackCandles, cR, df):
        return 1
    elif df.iloc[l]["Rejection"] == 2 and cR and is_above_support(l, windowsBackCandles, cS, df):
        return 2
    else:
        return 0




data = get_historical("ETHUSDT", "1d", "2023-01-01")
data = identify_rejection(data)

#--------------------------------
n1 = 8
n2 = 8
levelBackCandles = 60
windowsBackCandles = n2

signal = [0 for i in range(len(data))]

for row in tqdm(range(levelBackCandles+n1, len(data)-n2)):
    signal[row] = check_candle_signal(row, n1, n2, levelBackCandles, windowsBackCandles, data)

data["Signal"] = signal
data = data[data["Signal"]!=0]

print(data)