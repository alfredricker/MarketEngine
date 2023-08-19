import yfinance as yf
import pandas as pd
import requests
import json
from datetime import datetime,timedelta
import quandl

quandl.ApiConfig.api_key = 'h3gFTxuuELMsc6zXU7_J'

#usign alpha vantage again
def get_earnings(symbol:str):
    url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey=1C3O71BAB7HJXTWZ'
    r = requests.get(url)
    earnings_data = r.json()
    with open(f'data_equity/{symbol}_earnings.dat','w') as file:
        json.dump(earnings_data,file)
    return earnings_data

def get_balance_sheet(symbol:str):
    url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}&output_size=max&apikey=1C3O71BAB7HJXTWZ'
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


def get_cashflow(symbol:str):
    url = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={symbol}&time_from=20050101T0000&apikey=1C3O71BAB7HJXTWZ'
    r = requests.get(url)
    data = r.json()
    with open(f'data_equity/{symbol}_cashflow.dat','w') as file:
        json.dump(data,file)
    return data


#history only goes as far back as 2016. I will probel have to integrate this into a different neural network that also utilizes a news sentiment model.
def get_retail_sentiment_quandl(symbol:str,increment:int=20,start_date=datetime(2016,1,1),end_date=datetime(2023,6,1)):    
    #I have to make a list because the quandl api is annoying and calls by either a single date or an enumerated list of dates.
    #calling by each date individually takes a really long time so I have to go with this route
    def day_string(start_date,length:int):
        days_string = ', '.join([(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(length)])
        return days_string

    current_date = start_date

    while current_date+timedelta(days=increment) < end_date:
        if increment == 1:
            date_string = current_date.strftime('%Y-%m-%d')
        else:
            date_string = day_string(current_date,increment)
        data = quandl.get_table('NDAQ/RTAT10', date=date_string, ticker=symbol)
        data = pd.DataFrame(data)
        
        if current_date == start_date:
            df = data
        else:
            df = pd.concat([df,data],axis=0,ignore_index=True)
        current_date = current_date + timedelta(days=increment)
        print(current_date)
        df = pd.concat([df,data],axis=0,ignore_index=True)

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['date'],inplace=True)
    j = df.to_json(orient='records',date_format='iso')
    with open(f'data_equity/{symbol}_retail_sentiment.dat','w') as file:
        file.write(j)
    return j


#history only goes as far back as 2016. I will probel have to integrate this into a different neural network that also utilizes a news sentiment model.
def get_retail_sentiment(symbol:str,increment:int=20,start_date=datetime(2016,1,1),end_date=datetime(2023,6,1)):    
    #calling by each date individually takes a really long time so I have to go with this route
    def day_string(start_date,length:int):
        days_string = ', '.join([(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(length)])
        return days_string

    current_date = start_date
    next_date = current_date + timedelta(days=increment)

    while next_date < end_date:
        if increment == 1:
            date_string = current_date.strftime('%Y-%m-%d')
        else:
            date_string = day_string(current_date,increment)
        url = f'https://data.nasdaq.com/api/v3/datatables/NDAQ/RTAT10?date={date_string}&ticker={symbol}&api_key=h3gFTxuuELMsc6zXU7_J'
        data = requests.get(url)
        j = data.json()
        print(j)
        data = j['meta']
        #j = json.loads(j)
        data = pd.DataFrame(data)

        if current_date == start_date:
            df = data
        else:
            df = pd.concat([df,data],axis=0,ignore_index=True)
        current_date = next_date
        next_date = next_date + timedelta(days=increment)
        print(current_date)
    print(df)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['date'],inplace=True)
    j = df.to_json(orient='records',date_format='iso')
    with open(f'data_equity/{symbol}_retail_sentiment.dat','w') as file:
        file.write(j)
    return j


def balance_sheet_formatter(symbol:str):
    with open(f'data_equity/{symbol}_balance_sheet.dat','r') as file:
        f = file.read()
    j = json.loads(f)
    dat = j['quarterlyReports']
    df = pd.DataFrame(dat)
    df.rename(columns={'fiscalDateEnding': 'Date'},inplace=True)
    df.to_csv(f'csv_tests/{symbol}_balance_sheet.csv')


    
#get_balance_sheet(symbol)
#balance_sheet_formatter(symbol)
#get_earnings('WMT')
#get_retail_sentiment_quandl('UPS',increment=1,start_date=datetime(2020,1,1),end_date=datetime(2020,3,1))



