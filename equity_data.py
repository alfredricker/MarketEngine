import yfinance as yf
import pandas as pd
import requests
import json

#usign alpha vantage again
def get_earnings(symbol):
    url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey=1C3O71BAB7HJXTWZ'
    r = requests.get(url)
    earnings_data = r.json()
    with open(f'data_equity/{symbol}_earnings.dat','w') as file:
        json.dump(earnings_data,file)
    return earnings_data

def get_balance_sheet(symbol):
    url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}&apikey=1C3O71BAB7HJXTWZ'
    r = requests.get(url)
    data = r.json()
    with open(f'data_equity/{symbol}_balance_sheet.dat','w') as file:
        json.dump(data,file)
    return data

get_balance_sheet('AAPL')