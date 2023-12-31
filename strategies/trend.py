import pandas as pd
import pandas_ta as ta
import numpy as np

class Trader():
    def __init__(self, client, symbol, bar_length, start, stop_loss, take_profit, units):
        self.client = client
        self.symbol = symbol
        self.available_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
        self.bar_length = bar_length
        self.start = start
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.units = units
        self.entry_price = None
        self.in_position = False
        self.dynamic_stop_loss = None

    
    def start_trading(self):
         if self.bar_length in self.available_intervals:
                 self.get_historical(symbol = self.symbol, interval = self.bar_length, start = self.start)
        
         self.calculate_indicators()
         self.execute_trades()

    def get_historical(self, symbol, interval, start):
        df = pd.DataFrame(self.client.get_historical_klines(symbol=symbol, interval=interval, start_str=start))
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
    
    def update_dynamic_stop_loss(self):
        if self.in_position and self.entry_price and not self.dynamic_stop_loss:
            current_price = self.data["Close"].iloc[-1]
            profit_target = self.entry_price * (1 + self.take_profit / 2)

            if current_price >= profit_target:
                self.dynamic_stop_loss = self.entry_price

    def check_exit_conditions(self):
        last_close = self.data["Close"].iloc[-1]

        stop_loss_price = self.dynamic_stop_loss if self.dynamic_stop_loss else self.entry_price * (1 - self.stop_loss)
        if last_close <= stop_loss_price:
            return True
        
        if hasattr(self, 'entry_price'):
            stop_loss_price = self.entry_price * (1 - self.stop_loss)
            take_profit_price = self.entry_price * (1 + self.take_profit)
            

            if last_close <= stop_loss_price or last_close >= take_profit_price:
                return True
            
        if hasattr(self, 'data') and "SMA_10" in self.data.columns and "SMA_30" in self.data.columns:
            if self.data["SMA_10"].iloc[-1] < self.data["SMA_30"].iloc[-1]:
                return True


        return False
    
    def execute_trades(self):
        self.update_dynamic_stop_loss()
        entry = self.data["Confirmed Signal"].iloc[-1]

        try:
            if entry == 2 and not self.in_position:
                order = self.client.create_order(symbol=self.symbol, side="BUY", type="MARKET", quantity=self.units)
                self.entry_price = self.data["Close"].iloc[-1]
                print(f"Achat effectué : {order}")
                self.in_position = True
            elif self.in_position and self.check_exit_conditions():
                order = self.client.create_order(symbol=self.symbol, side="SELL", type="MARKET", quantity=self.units)
                self.entry_price = None
                print(f"Vente effectuée : {order}")
                self.in_position = False
                self.dynamic_stop_loss = None
            else:
                print(f"Aucune transaction effectué")
            
        except Exception as e:
                print(f"Erreur: {e}")
    

def run_trend_strategy(client, symbol, bar_length, start, stop_loss, take_profit, units):
    trader = Trader(client, symbol, bar_length, start, stop_loss, take_profit, units)
    trader.start_trading()
    return trader.data[trader.data["Confirmed Signal"] != 0]
