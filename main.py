import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
import os
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
from binance import ThreadedWebsocketManager
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

load_dotenv()
api_key = os.getenv("API_KEY_TEST")
secret_key = os.getenv("SECRET_KEY_TEST")

# Binance Client
client = Client(api_key=api_key, api_secret=secret_key, tld='com', testnet=False)



# Créer le graphique à chandeliers
# fig = go.Figure(df=[go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"])])
#         # Ajouter des lignes pour les indicateurs
# fig.add_trace(go.Scatter(x=df.index, y=df["EMA_Short"], mode="lines", name="EMA_Short", line=dict(color="blue")))
# fig.add_trace(go.Scatter(x=df.index, y=df["EMA_Long"], mode="lines", name="EMA_Long", line=dict(color="crimson")))
# fig.add_trace(go.Scatter(x=df[df["Entry"] == 1].index, y=df[df["Entry"] == 1]["Close"], 
#                          mode="markers", marker_symbol="triangle-up", marker_color="green",marker_size=15, name="Buy"))
# fig.add_trace(go.Scatter(x=df[df["Entry"] == -1].index, y=df[df["Entry"] == -1]["Close"], 
#                          mode="markers", marker_symbol="triangle-down", marker_color="red",marker_size=15, name="Sell"))

      
# Afficher le graphique
# fig.show()


#----------------------------------
# Implémentation de la class
class Trader():
      def __init__(self, symbol, bar_length, stop_loss, target_profit, units):
            self.symbol = symbol
            self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
            self.bar_length = bar_length
            self.stop_loss = stop_loss
            self.target_profit = target_profit
            self.units = units
            self.fast_length = 12
            self.slow_length = 26
            self.signal_length = 9

        
      def start_trading(self):
            self.twm = ThreadedWebsocketManager()
            self.twm.start()

            if self.bar_length in self.available_intervals:
                 self.get_recent(symbol = self.symbol, interval = self.bar_length)
                 self.twm.start_kline_socket(callback = self.stream_candles,
                                        symbol = self.symbol, interval = self.bar_length)
            
            self.calculate_indicator()
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
        
     
    
      # Generate all trading's indicators
      def calculate_indicator(self):
        # Calcul des EMAs
        self.data['EMA_Short'] = ta.ema(self.data["Close"], 20)
        self.data['EMA_Long'] = ta.ema(self.data["Close"], 50)

        # Calculate MACD
        macd = ta.macd(self.data["Close"], fast=self.fast_length, slow=self.slow_length, signal=self.signal_length)
        self.data['MACD'] = macd['MACD_12_26_9']
        self.data['MACD_Histogram'] = macd['MACDh_12_26_9']
        self.data['MACD_Signal'] = macd['MACDs_12_26_9']
        self.generate_crossover_signal()
        self.generate_macd_signal()
        self.generate_entry_signal()
      
      
      # Generate CROSSOVER signal
      def generate_crossover_signal(self):
        self.data["EMA_Trend"] = 0.0
        self.data["EMA_Trend"] = np.where(self.data["EMA_Short"] > self.data["EMA_Long"], 1.0, 0.0)
        self.data["Crossover_Trend"] = self.data["EMA_Trend"].diff()
      
      # Generate MACD signal
      def generate_macd_signal(self):
         macd_bullish = (self.data['MACD'] > self.data['MACD_Signal'])
         macd_bearish = (self.data['MACD'] < self.data['MACD_Signal'])

         self.data["MACD_Trend"] = np.where(macd_bullish, 1, np.where(macd_bearish, -1, 0))
      
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

        # Vérifier si la nouvelle bougie est complète
        if complete:
            if start_time in self.data.index:
                # Mettez à jour la bougie existante
                self.data.loc[start_time] = new_data.loc[start_time]
            else:
                # Concaténer avec les données existantes
                self.data = pd.concat([self.data, new_data])

      # Combine CROSSOVER signal with MACD signal for generate a entry signal
      def generate_entry_signal(self):
         buy_signal = (self.data["Crossover_Trend"] > 0) & (self.data["MACD_Trend"] == 1) # Buy signal
         sell_signal = (self.data["Crossover_Trend"] < 0) & (self.data["MACD_Trend"] == -1) # Sell signal

         self.data["Entry"] = np.where(buy_signal, 1, np.where(sell_signal, -1, 0))
       

      def execute_trades(self):
        entry = self.data["Entry"].iloc[-1]

        try:
            if entry == 1:
                # order = client.create_order(symbol=self.symbol, side="BUY", type="MARKET", quantity=self.units)
                print(f"Achat effectué : ")
            elif entry == -1:
                # order = client.create_order(symbol=self.symbol, side="SELL", type="MARKET", quantity=self.units)
                print(f"Vente effectuée : ")
            else:
                print(f"neutre")
        except Exception as e:
                print(f"Erreur: {e}")

     

 # Variables de trading
bar_length = "1h"
symbol = "BTCUSDT"
pourcentage_risque_par_trade = 0.01
target_profit = 1.01
stop_loss = 0.99
units = 10

# Instance de la class Trader
trader = Trader(symbol=symbol, bar_length=bar_length, stop_loss=stop_loss, target_profit=target_profit, units=units)

trader.start_trading()
run_time = 30 
start_time = time.time()
while time.time() - start_time < run_time:
    time.sleep(1)

trader.twm.stop()

data = trader.data[trader.data["Entry"]!=0]
# data = trader.data
print(data)
            
          