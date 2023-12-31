from strategies.trend import run_trend_strategy
import os
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY_TEST")
secret_key = os.getenv("SECRET_KEY_TEST")
client = Client(api_key=api_key, api_secret=secret_key, tld='com', testnet=True)

def main():
    # Strategy variables
    symbol = "RNDRUSDT"
    bar_length = "1h"
    start = "2023-11-01"
    
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
    pourcentage_risque_par_trade = 0.01
    montant_risque = capital * pourcentage_risque_par_trade

    # Units to trade
    precision = 5
    unit = montant_risque / btc_price
    units = round(unit, precision)

    # Stop Loss and Target Profit
    ratio_risque_rendement = 3
    pourcentage_stop_loss = 0.05
    pourcentage_take_profit = pourcentage_stop_loss * ratio_risque_rendement
    stop_loss = 1 - pourcentage_stop_loss 
    take_profit = 1 + pourcentage_take_profit

    # Exécution de la stratégie
    data = run_trend_strategy(client, symbol, bar_length, start, stop_loss, take_profit, units)
    print(data)

# Exécution de la fonction principale
if __name__ == "__main__":
    main()