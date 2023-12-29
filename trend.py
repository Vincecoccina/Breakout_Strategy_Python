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


class Trader():
    def __init__(self, symbol, bar_length, start, stop_loss, take_profit, units):
        self.symbol = symbol
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        self.bar_length = bar_length
        self.start = start
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.units = units
    
    def start_trading(self):
         if self.bar_length in self.available_intervals:
                 self.get_historical(symbol = self.symbol, interval = self.bar_length, start = self.start)
        
         self.calculate_indicators()

    def get_historical(self, symbol, interval, start):
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
        self.data = df
    
    # Calculate SMA
    def calculate_sma(self, length):
         return self.data.ta.sma(length=length)
    
    def calculate_adx(self):
        adx_data = self.data.ta.adx()
        adx_data = adx_data.rename(columns=lambda x: x[:3] if x.startswith("ADX") or x.startswith("DM") else x)
        self.data = self.data.join(adx_data)
    
    # Calculate Trend Direction
    def calculate_slope(self, series, period=5):
        slopes = [0 for _ in range(period-1)]
        for i in range(period-1, len(series)):
            x = np.arange(period)
            y = series.iloc[i-period+1:i+1].values
            slope = np.polyfit(x, y, 1)[0]
            percent_slope = (slope / y[0]) * 100 if y[0] != 0 else 0 
            slopes.append(percent_slope)
        return slopes
    
    # Calculate Trend Signal
    def determinate_trend(self, row):
        if row["SMA_10"] > row["SMA_20"] > row["SMA_30"]:
            return 2 # Up Trend
        elif row["SMA_10"] < row["SMA_20"] < row["SMA_30"]:
            return 1 # Down Trend
        else:
            return 0 # No Trend
    
    def check_candles(self, backcandles, ma_column):
        categories = [0 for _ in range(backcandles)]
        for i in range(backcandles, len(self.data)):
            if all(self.data.Close[i-backcandles:i] > self.data[ma_column][i-backcandles:i]):
                categories.append(2) # Up Trend
            elif all(self.data.Close[i-backcandles:i] < self.data[ma_column][i-backcandles:i]):
                categories.append(1) # Down Trend
            else:
                categories.append(0)

        return categories
    
    def generate_trend_signal(self, threshold=40):
        trend_signal = []
        for i in range(len(self.data)):
            if self.data["ADX"].iloc[i] > threshold:
                if self.data["DMP"].iloc[i] > self.data["DMN"].iloc[i]:
                    trend_signal.append(2) # Up Trend
                else:
                    trend_signal.append(1) # Down Trend
            else:
                trend_signal.append(0) # No Trend
        return trend_signal
    
    def calculate_indicators(self):
         self.data["SMA_10"] = self.calculate_sma(10)
         self.data["SMA_20"] = self.calculate_sma(20)
         self.data["SMA_30"] = self.calculate_sma(30)
         self.calculate_adx()
         self.data["Slope"] = self.calculate_slope(self.data["SMA_20"])
         self.data["Trend"] = self.data.apply(self.determinate_trend, axis=1)
         self.data["Category"] = self.check_candles(5, "SMA_20")
         self.data["Trend Signal"] = self.generate_trend_signal()
         self.data["Confirmed Signal"] = self.data.apply(lambda row: row["Category"] if row["Category"] == row["Trend Signal"] else 0, axis=1)
         self.data.dropna(inplace=True)


if __name__ == "__main__":

    api_key = os.getenv("API_KEY_TEST")
    secret_key = os.getenv("SECRET_KEY_TEST")

    # Binance Client
    client = Client(api_key=api_key, api_secret=secret_key, tld='com', testnet=False)

    symbol = "BTCUSDT"
    bar_length = "1d"
    start = "2023-09-01"
    stop_loss = 0.95 
    take_profit = 1.05
    units = 10
    trader = Trader(symbol=symbol, bar_length=bar_length, start=start, stop_loss=stop_loss, take_profit=take_profit, units=units)

    trader.start_trading()
    data = trader.data
    print(data)

    


