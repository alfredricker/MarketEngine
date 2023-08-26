#THIS FILE COMPILES ALL OF THE RELEVANT EQUITY DATA THAT YOU WOULD LIKE TO PROCESS INTO ONE DATAFRAME THEN EXPORTS IT TO data.csv
#This file also compiles all of the sentiment data and exports it to SENTIMENT.csv
import pandas as pd
import json
import functions as fn
from datetime import datetime

#MODEL STOCKS
#stocks currently in the model. It is import that these two lists are aligned ticker:company
retail_stocks = ['AAPL','AMD','AMZN','MSFT','NVDA','TSLA']
retail_companies = ['Apple','AMD','Amazon','Microsoft','Nvidia','Tesla']
#TREND DATA
def trend_init(symbol:str):
    with open(f'data_misc/trend_{symbol}.dat', 'r') as file:
        data = file.read()
    trend_df = pd.read_json(data)
    return trend_df

#FED DATA
def fed_init(code:str):
    with open(f'data_fed/{code}_formatted.dat','r') as file:
        r = file.read()
    df = pd.read_json(r)
    return df

gdp = 'gdp'
cpi = 'cpiaucsl'
dff = 'dff'
m1 = 'm1'
m1v = 'm1v'
indpro = 'indpro'
unrate = 'unrate'
gdp = fed_init(gdp)
cpi = fed_init(cpi)
dff = fed_init(indpro)
m1 = fed_init(m1)
m1v = fed_init(m1v)
indpro = fed_init(indpro)
unrate = fed_init(unrate)
print("Initialized fed data")

#HOUSING DATA
def housing_init(city:str):
    with open(f'data_misc/{city}_formatted.dat','r') as file:
        r = file.read()
    data = pd.read_json(r)
    return data
ny = housing_init('NewYork')
la = housing_init('LosAngeles')
chicago = housing_init('Chicago')
print("Initialized housing data")

def insider_init(symbol:str):
    with open(f'data_misc/insider_{symbol}.dat','r') as file:
        r = file.read()
    data = pd.read_json(r,convert_axes=True)
    #replace all rows where insider_flow==1 to =0
    data['InsiderFlow'] = data['InsiderFlow'].replace(1,0)
    df = fn.multiple_date_fill(data)
    return df

#EQUITY/TARGET DATA
#surprise column contains a NaN value which interrupts the model as of now.
def equity_init(stocks_list):
    df_list = [] #list of data frames that will be concatenated by row at the end of the function

    for i in range(len(stocks_list)):    
        stock_sym = stocks_list[i]
        print(f'Initializing {stocks_list[i]} data...')
        stock = fn.equity_formatter(stock_sym)
        stock_close = stock[0]
        stock_volume = stock[1]
        stock_rsi = fn.calculate_rsi(stock_sym)
        
        stock_trend = trend_init(stock_sym)
        stock_earnings = fn.earnings_formatter(stock_sym)
        stock_reported_eps = stock_earnings[0]
        stock_surprise_eps = stock_earnings[1]

        insider = insider_init(stock_sym)
        retail_df = fn.retail_sentiment_formatter(stock_sym)
        news_sentiment = news_init(stock_sym)
        alpha = news_init(stock_sym,prefix='seekingalpha')

        #CONCATENATE ALL DATA INTO A SINGLE LARGE DATAFRAME
        data_list = [stock_close,stock_volume,stock_trend,stock_rsi,stock_reported_eps,stock_surprise_eps,
                     gdp,cpi,dff,m1,unrate,ny,insider,retail_df,news_sentiment,alpha]
        df = fn.concatenate_data(data_list)
        df['Close_Tmr'] = df['Close'].shift(-1)
        df = df.drop(index=df.index[-1]) #drop the last row because the above shift method will result in NaN values
        df.set_index('Date',inplace=True)
        df_list.append(df)
    if len(df_list) < 2:
        data = df_list[0]
    else:    
        data = pd.concat(df_list,axis=0,ignore_index=True)
    data.to_csv(f'csv_data/{stocks_list[0]}.csv')
    print(f'Initialized equity data')
    return data
#IMPORTANT: since I want to look at how the present day data effects the closing price of tomorrow, I have to copy the close column, and shift it up 1 position in a new row


def news_init(symbol,prefix='news_sentiment'):
    with open(f'data_misc/{prefix}_{symbol}.dat','r') as file:
        r = file.read()
    df = pd.read_json(r)
    return df


def sentiment_init(stocks_list):
    df_list = []    
    for i in range(len(stocks_list)):
        sym = stocks_list[i]
        news_sentiment = news_init(sym)
        alpha = news_init(sym,prefix='seekingalpha')
        stock_trend = trend_init(sym)

        stock = fn.equity_formatter(sym)
        stock_close = stock[0]
        stock_volume = stock[1]
        stock_rsi = fn.calculate_rsi(sym)

        insider = insider_init(sym)
        retail_df = fn.retail_sentiment_formatter(sym)

        stock_earnings = fn.earnings_formatter(sym)
        stock_reported_eps = stock_earnings[0]
        stock_surprise_eps = stock_earnings[1]

        data_list = [stock_close,stock_volume,stock_reported_eps,stock_surprise_eps,
                     stock_trend,news_sentiment,alpha,stock_rsi,insider,retail_df,dff]
        df = fn.concatenate_data(data_list)
        df['Close_Tmr'] = df['Close'].shift(-1)
        df = df.drop(index=df.index[-1]) #drop the last row because the above shift method will result in NaN values
        df.set_index('Date',inplace=True)
        print(f'Finished formatting {sym}')
        df_list.append(df)

    data = pd.concat(df_list,axis=0,ignore_index=True)
    data.to_csv('csv_data/CLASSIFIER_1.csv')
    print(f'Initialized sentiment data')
    return data


#sentiment_init(retail_stocks,retail_companies)
#equity_init(model_stocks,model_companies)
#as of now, seekingalpha_F.dat ends in 2015, so I am leaving ford out of the list
model_stocks = ['AAL','AAPL','AMD','AMZN','BAC','BANC','BRK-B','CGNX','UPS','DELL','DIS',
                'INTC','GE','TGT','MCD','MSFT','NVDA','NFLX','QCOM','ROKU','RUN','SBUX','WMT','GOOG','CSCO']
sentiment_init(model_stocks)
