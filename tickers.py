import json
import requests
import yfinance as yf #i think this is defunct
import googlefinance as gf #i think this is also defunct
import pandas as pd
import torch
from datetime import datetime, timedelta

#gets closing prices from yfinance
def get_equity_data(symbol,start_date='2000-01-01',end_date='2023-06-01'):
    data = yf.download(symbol,start=start_date,end=end_date)
    df = pd.DataFrame(data)
    df.reset_index(inplace=True)
    j = df.to_json(orient='records',date_format='iso')
    with open(f'data_equity/{symbol}.dat', 'w') as file:
        file.write(j)

get_equity_data('MCD')


#create a list of tickers from csv data frame
'''
df = pd.read_csv("nasdaq_screener.csv")
df_sorted = df.sort_values(by='Volume',ascending=False)
n = 500
#reduce the list to the tickers with the n highest volume
ticker_list = [s for s in df_sorted.head(n)['Symbol']]

for i in range(36,41):
    url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=" + ticker_list[i] + "&outputsize=full&apikey=1C3O71BAB7HJXTWZ"
    try:
        r = requests.get(url)
        data = r.json()
        if 'Error Message' in data:
            print("Error in alphavantage request for " ,ticker_list[i])
        else:
            with open(f'data_equity/{ticker_list[i]}.dat','w') as file:
                json.dump(data,file)
                
    except:
        print("An error occured: could not request AlphaVantage data")

for tick in ticker_list:
    url="https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=" + tick + "&outputsize=full&apikey=1C3O71BAB7HJXTWZ"
    r=requests.get(url)
    data = r.json()
'''
