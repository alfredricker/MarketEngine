#GOOGLE SEARCH TREND DATA
import pytrends
import pandas as pd
import torch
import requests
import json
import datetime
from datetime import timedelta
import functions as fn

from pytrends.request import TrendReq
pytrends = TrendReq(hl='en-US',tz=360)

def form_two_columns(row):
    row_str = str(row)
    row_list = row_str.split('\n',2)
    value = float(row_list[0].split(' ',1)[1])
    date = (row_list[2].split(' ',2)[1])
    return [date,value]


#input the company name and the stock symbol of that company
#I'll make the start date june 2004
def get_search_trend_data(company):
    kw_list = [company]
    start_date = datetime.datetime(2004,6,1)
    end_date = datetime.datetime(2022,6,1)
    current_date = start_date
    dataframe = {'Date':[],'Value':[]}

    while current_date<end_date:
        inc_date = current_date+timedelta(days=90)
        current_str = current_date.strftime('%Y-%m-%d')
        inc_str = inc_date.strftime('%Y-%m-%d')
        #print(start_date)
        time_str = f'{current_str} {inc_str}'
        interest = pytrends.build_payload(kw_list,cat=12,timeframe=time_str,geo='',gprop='') #cat=12 for business and industrial category
        interest_df = pytrends.interest_over_time()
        for index,value_row in interest_df.iterrows():
            arr = form_two_columns(value_row)
            dataframe['Date'].append(arr[0])
            dataframe['Value'].append(arr[1])   
        current_date=inc_date+timedelta(days=1)

    df = pd.DataFrame(dataframe)
    trend_json = df.to_json(orient='records',date_format='iso')
    trend_json.strip('[]')
    trend_load = json.loads(trend_json)
    with open(f'data_misc/trend_{company}.dat','w') as file:
        json.dump(trend_load,file)
    return trend_load


#don't need a formatted the data is already clean

#create list of 10 most popular tickers from a pandas dataframe
df = pd.read_csv("nasdaq_screener.csv")
df_sorted = df.sort_values(by='Volume',ascending=False)
n = 10
#reduce the list to the tickers with the n highest volume
company_list = [s for s in df_sorted.head(n)['Name']]
#print(company_list)
#get_search_trend_data('TUP')