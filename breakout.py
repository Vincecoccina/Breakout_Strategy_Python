import pandas as pd
import pandas_ta as ta
import numpy as np
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
      def __init__(self, symbol, bar_length, stop_loss, target_profit, units):
            self.symbol = symbol
            self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
            self.bar_length = bar_length
            self.stop_loss = stop_loss
            self.target_profit = target_profit
            self.units = units
            self.lookback_periods = 30
            self.in_position = False

        
      def start_trading(self):
            self.twm = ThreadedWebsocketManager()
            self.twm.start()

            if self.bar_length in self.available_intervals:
                 self.get_recent(symbol = self.symbol, interval = self.bar_length)
                 self.twm.start_kline_socket(callback = self.stream_candles,
                                        symbol = self.symbol, interval = self.bar_length)
            
            self.execute_trades()

      # Get Historical data
      def get_recent(self, symbol, interval):
        start = "2023-01-01"

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
        self.calculate_indicators()
        
         
      # Generate all trading's indicators
      def calculate_indicators(self):
        self.data['EMA'] = ta.ema(self.data.Close, length=150)
        self.data.dropna(inplace=True)
        self.data.tail()
        self.calculate_ema_signal()
    
      def calculate_ema_signal(self):
        EMAsignal = [0] * len(self.data)
        backcandles = 15

        for row in range(backcandles, len(self.data)):
            upt = 1
            dnt = 1
            for i in range(row - backcandles, row + 1):
                if max(self.data.iloc[i].Open, self.data.iloc[i].Close) >= self.data.iloc[i].EMA:
                    dnt = 0
                if min(self.data.iloc[i].Open, self.data.iloc[i].Close) <= self.data.iloc[i].EMA:
                    upt = 0
            if upt == 1 and dnt == 1:
                EMAsignal[row] = 3
            elif upt == 1:
                EMAsignal[row] = 2
            elif dnt == 1:
                EMAsignal[row] = 1

        self.data['EMASignal'] = EMAsignal
        
                   
      def stream_candles(self, msg):
        event_time = pd.to_datetime(msg["E"], unit = "ms")
        start_time = pd.to_datetime(msg["k"]["t"], unit = "ms")
        first   = float(msg["k"]["o"])
        high    = float(msg["k"]["h"])
        low     = float(msg["k"]["l"])
        close   = float(msg["k"]["c"])
        volume  = float(msg["k"]["v"])
        complete = msg["k"]["x"]
        print("Time: {} | Price: {}".format(event_time, close))
        new_data = pd.DataFrame({
        "Open": [first],
        "High": [high],
        "Low": [low],
        "Close": [close],
        "Volume": [volume],
        "Complete": [complete]
        }, index=[start_time])

        # Check if the candle is complete
        if complete:
            if start_time in self.data.index:
                # Update candle
                self.data.loc[start_time] = new_data.loc[start_time]
            else:
                # Concat with existing data
                self.data = pd.concat([self.data, new_data])
            self.calculate_indicators()
      
      def execute_trades(self):
          pass
      
     
if __name__ == "__main__":

    api_key = os.getenv("API_KEY_TEST")
    secret_key = os.getenv("SECRET_KEY_TEST")

    # Binance Client
    client = Client(api_key=api_key, api_secret=secret_key, tld='com', testnet=True)

    # Symbol and Interval variables
    symbol = "ETHUSDT"
    bar_length = "1h"

    # Get account data
    account_info = client.get_account()
    btc_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
    balances = account_info['balances']
    for balance in balances:
        if balance['asset'] == 'USDT':
            usdt_balance = float(balance['free'])
            print("Solde en USDT : ", usdt_balance)

    # Trading Variables           
    capital = usdt_balance
    price = btc_price
    pourcentage_risque_par_trade = 0.01
    montant_risque = capital * pourcentage_risque_par_trade

    # Units to trade
    precision = 5
    unit = montant_risque / btc_price
    units = round(unit, precision)

    # Stop Loss and Target Profit
    ratio_risque_rendement = 3
    pourcentage_stop_loss = 0.05
    pourcentage_target_profit = pourcentage_stop_loss * ratio_risque_rendement
    stop_loss = 1 - pourcentage_stop_loss 
    target_profit = 1 + pourcentage_target_profit

    # Trader Instance
    trader = Trader(symbol=symbol, bar_length=bar_length, stop_loss=stop_loss, target_profit=target_profit, units=units)

    trader.start_trading()
    run_time = 60 
    start_time = time.time()
    while time.time() - start_time < run_time:
        time.sleep(1)

    trader.twm.stop()

    data = trader.data[trader.data["EMASignal"]!=0]
    print(data)

#---------------------
# Implémenter Backtesting
    
            
          