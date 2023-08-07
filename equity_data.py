import yfinance as yf
import pandas as pd
import requests
import json

#usign alpha vantage again
def get_earnings(symbol:str):
    url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey=1C3O71BAB7HJXTWZ'
    r = requests.get(url)
    earnings_data = r.json()
    with open(f'data_equity/{symbol}_earnings.dat','w') as file:
        json.dump(earnings_data,file)
    return earnings_data

def get_balance_sheet(symbol:str):
    url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}&apikey=1C3O71BAB7HJXTWZ'
    r = requests.get(url)
    data = r.json()
    with open(f'data_equity/{symbol}_balance_sheet.dat','w') as file:
        json.dump(data,file)
    return data

def get_news_sentiment(symbol:str):
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&time_from=20050101T0000&apikey=1C3O71BAB7HJXTWZ'
    r = requests.get(url)
    data = r.json()
    with open(f'data_equity/{symbol}_news_sentitment.dat','w') as file:
        json.dump(data,file)
    return data

symbol = 'AAPL'
#get_balance_sheet(symbol)
get_news_sentiment(symbol)

def balance_sheet_formatter(symbol:str):
    with open(f'data_equity/{symbol}_balance_sheet.dat','r') as file:
        f = file.read()
    j = json.loads(f)
    dat = j['quarterlyReports']
    df = pd.DataFrame(dat)
    df.rename(columns={'fiscalDateEnding': 'Date'},inplace=True)
    df.to_csv(f'csv_tests/{symbol}_balance_sheet.csv')
    
#balance_sheet_formatter(symbol)
