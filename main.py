from strategies.trend import run_trend_strategy
import numpy as np
import os
from binance.client import Client
import datetime
import threading
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")
secret_key = os.getenv("SECRET_KEY")
client = Client(api_key=api_key, api_secret=secret_key, tld='com')


def adjust_units(client, symbol, desired_units):
    exchange_info = client.get_exchange_info()
    symbol_info = next((item for item in exchange_info['symbols'] if item['symbol'] == symbol), None)

    # Trouver les restrictions de lot pour le symbol
    lot_size_filter = next((filter for filter in symbol_info['filters'] if filter['filterType'] == 'LOT_SIZE'), None)
    min_qty = float(lot_size_filter['minQty'])
    step_size = float(lot_size_filter['stepSize'])

    # Calculer le nombre de décimales autorisé pour la quantité
    qty_precision = int(round(-np.log10(step_size)))

    # Ajuster les units pour respecter le step size et la précision
    adjusted_units = max(min_qty, round(desired_units / step_size) * step_size)
    adjusted_units = round(adjusted_units, qty_precision)  # Arrondir à la précision autorisée

    return adjusted_units



def main():
    # Liste des tokens à trader
    symbols = ["SOLUSDT", "ETHUSDT", "BTCUSDT"]
    bar_length = "1h"
    start = datetime.datetime.now() - datetime.timedelta(days=60)
    start = start.strftime("%Y-%m-%d %H:%M:%S")

    # Get account data
    account_info = client.get_account()
    usdt_balance = next((float(balance['free']) for balance in account_info['balances'] if balance['asset'] == 'USDT'), 0)
    print("Solde en USDT : ", usdt_balance)
    
    # Trading Variables
    capital = usdt_balance
    pourcentage_risque_par_trade = 0.05

    threads = []
    for symbol in symbols:
        token_price = float(client.get_symbol_ticker(symbol=symbol)['price'])
        montant_risque = capital * pourcentage_risque_par_trade

        # Units to trade
        desired_units = montant_risque / token_price
        units = adjust_units(client, symbol, desired_units)

        # Stop Loss and Target Profit
        ratio_risque_rendement = 1
        pourcentage_stop_loss = 0.02
        pourcentage_take_profit = pourcentage_stop_loss * ratio_risque_rendement
        stop_loss = 1 - pourcentage_stop_loss 
        take_profit = 1 + pourcentage_take_profit

        # Créer un thread pour chaque stratégie de token
        thread = threading.Thread(target=run_trend_strategy, args=(client, symbol, bar_length, start, stop_loss, take_profit, units))
        threads.append(thread)
        thread.start()

    # Attendre que tous les threads se terminent
    for thread in threads:
        thread.join()


if __name__ == "__main__":
     main()
