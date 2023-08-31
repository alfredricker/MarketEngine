#THIS FILE COMPILES ALL OF THE RELEVANT EQUITY DATA THAT YOU WOULD LIKE TO PROCESS INTO ONE DATAFRAME THEN EXPORTS IT TO data.csv
#This file also compiles all of the sentiment data and exports it to SENTIMENT.csv
import pandas as pd
import numpy as np
import json
import requests
import functions as fn
from datetime import datetime,timedelta
from bs4 import BeautifulSoup
import yfinance as yf
import quandl
import time
from selenium import webdriver
import subprocess

ndaq = pd.read_csv("csv_data/nasdaq_screener.csv")
ndaq = ndaq.sort_values(by='Volume',ascending=False)

quandl.ApiConfig.api_key = 'h3gFTxuuELMsc6zXU7_J'

def data_appender(pth:str,
                  get_function:callable, #the function that fetches the data
                  symbol:str, #the string passed to the get function
                  start_date:datetime=None,
                  end_date:datetime=datetime.now(),
                  concat:bool=True, #concat will be false for alphavantage data (because start and end date do nothing within those functions)
                  overwrite = True):  
    with open(pth,'r') as f:
        r = f.read()


    df = pd.read_json(r,convert_dates=True)
    if "seekingalpha" in pth: #this is temporary because i didn't originally include this line in my seekingalpha function
        df['Date'] = df['Date'].apply(lambda x: x[:10])
        df.loc[:,'Date'] = pd.to_datetime(df['Date'])
    
    df.sort_values(by='Date',inplace=True)
    
    recent_date = df['Date'].iloc[-1]
    #print(recent_date)
    if recent_date+timedelta(days=1)>end_date:
        print('No new data to append')
        start_date = fn.find_closest_date(df,start_date,direction='left')
        df = df.loc[df['Date']>=start_date]
        return df

    if start_date is None:
        starting_date = recent_date#datetime.strptime(str(recent_date),'%Y-%m-%d')
        starting_date = starting_date + timedelta(days=1)
    elif start_date<recent_date:
        starting_date = recent_date
        starting_date = starting_date + timedelta(days=1)
    else:
        starting_date=start_date
    
    #print(starting_date,'\n',end_date)
    data = get_function(symbol,start_date=starting_date,end_date=end_date,file_method=None)
    #print(data)

    if concat==True:
        df = pd.concat([df,data],axis=0,ignore_index=True)
    else:
        df = data
    
    #print(df.to_string())
    j = df.to_json(orient='records',date_format='iso')
    if overwrite==True:
        with open(pth,'w') as f:
            f.write(j)

    #df = df.loc[df['Date']>=start_date]

    return df






#--------------------------------------------------------------
#EQUITY DATA
#--------------------------------------------------------------
#gets closing prices from yfinance
def get_equity_data(symbol:str,start_date=datetime(2000,1,1),end_date=datetime(2023,6,1),api:str='yf',file_method='w'):
    start_date=datetime.strftime(start_date,'%Y-%m-%d')
    end_date=datetime.strftime(end_date,'%Y-%m-%d')

    if api == 'yf':
        data = yf.download(symbol,start=start_date,end=end_date)
        df = pd.DataFrame(data)
        df.reset_index(inplace=True)
        j = df.to_json(orient='records',date_format='iso')

        if file_method is not None: 
            with open(f'data_equity/{symbol}.dat', file_method) as file:
                file.write(j)

    elif api=='alphavantage':
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey=1C3O71BAB7HJXTWZ"
        try:
            r = requests.get(url)
            data = r.json()
            df = pd.DataFrame(data).reset_index()

            if 'Error Message' in data:
                print(f"Error in alphavantage request for {symbol}")
            elif file_method is not None:
                with open(f'data_equity/{symbol}_AV.dat',file_method) as file:
                    json.dump(data,file)             
        except:
            print("An error occured: could not request AlphaVantage data")
    
    else:
        print("Invalid api")
    return df


#create a list of tickers from csv data frame
def get_ticker_list(n:int=500,sort_by='Volume'):
    df = pd.read_csv("nasdaq_screener.csv")
    df_sorted = df.sort_values(by=sort_by,ascending=False)
    #reduce the list to the tickers with the n highest volume
    ticker_list = [s for s in df_sorted.head(n)['Symbol']]
    return ticker_list


#using alpha vantage again
def get_earnings(symbol:str,start_date=None,end_date=None,file_method='w'):
    url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey=1C3O71BAB7HJXTWZ'
    r = requests.get(url)
    j = r.json()
    dat = j['quarterlyEarnings']
    df = pd.DataFrame(dat)

    df = df[['reportedDate','reportedEPS','surprisePercentage']]
    #change reportedDate column to Date
    df.rename(columns={'reportedDate':'Date','surprisePercentage':'surprise'},inplace=True)
    df.loc[:,'Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date')

    if start_date is not None:
        df = df.loc[df['Date']>=start_date]
    if end_date is not None:
        df = df.loc[df['Date']<=end_date]

    data = df.to_json(orient='records',date_format='iso')

    if file_method is not None:
        with open(f'data_equity/{symbol}_earnings.dat',file_method) as file:
            file.write(data)
    return df



def earnings_init_old(symbol:str,start_date=None,end_date=None):
    with open(f'data_equity/{symbol}_earnings.dat','r') as file:
        r = file.read()
    data = pd.read_json(r,convert_dates=True)
    df = fn.data_fill(data,damping_columns=['surprise'],start_date=start_date,end_date=end_date)
    return df



def get_balance_sheet(symbol:str,start_date=None,end_date=None,file_method='w'):
    url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}&output_size=max&apikey=1C3O71BAB7HJXTWZ'
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data)

    if file_method is not None:
        with open(f'data_equity/{symbol}_balance_sheet.dat',file_method) as file:
            json.dump(data,file)
    return df


def get_cashflow(symbol:str,start_date=None,end_date=None,file_method='w'):
    url = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={symbol}&time_from=20050101T0000&apikey=1C3O71BAB7HJXTWZ'
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data)

    if file_method is not None:
        with open(f'data_equity/{symbol}_cashflow.dat',file_method) as file:
            json.dump(data,file)
    return df


#history only goes as far back as 2016. I will probel have to integrate this into a different neural network that also utilizes a news sentiment model.
def get_retail_sentiment(symbol:str,increment:int=20,start_date=datetime(2016,1,1),end_date=datetime(2023,6,1),method:str='quandl',file_method='w'):    
    #I have to make a list because the quandl api is annoying and calls by either a single date or an enumerated list of dates.
    #calling by each date individually takes a really long time so I have to go with this route
    def day_string(date_range):
        return ','.join([date.strftime('%Y-%m-%d') for date in date_range])

    current_date = start_date
    
    if symbol=='BRK-B':
        symbol = 'BRK.B'

    while current_date <= end_date:  # Updated loop condition

        # Generate a date range that covers the increment and ends on or before the end_date
        date_range = [current_date + timedelta(days=i) for i in range(min(increment, (end_date - current_date).days + 1))]

        date_string = day_string(date_range)
        
        if method == 'quandl':
            data = quandl.get_table('NDAQ/RTAT', date=date_string, ticker=symbol)
            data = pd.DataFrame(data)
        elif method=='url':
            url = f'https://data.nasdaq.com/api/v3/datatables/NDAQ/RTAT?date={date_string}&ticker={symbol}&api_key=h3gFTxuuELMsc6zXU7_J'
            data = requests.get(url)
            j = data.json()
            print(j)
            data = j['meta']
            #j = json.loads(j)
            data = pd.DataFrame(data)
        else:
            print('Error: valid methods are quandl,url')
            return 0
        
        if current_date == start_date:
            df = data
        else:
            df = pd.concat([df,data],axis=0,ignore_index=True)

        current_date = date_range[-1] + timedelta(days=1)  # Move to the next date
        #print(current_date)
        #df = pd.concat([df,data],axis=0,ignore_index=True)

    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by=['date'],inplace=True)
    df.rename(columns={'date':'Date'},inplace=True)

    j = df.to_json(orient='records',date_format='iso')

    if symbol=='BRK.B':
        symbol='BRK-B'

    if file_method is not None:
        with open(f'data_equity/{symbol}_retail_sentiment.dat',file_method) as file:
            file.write(j)

    print(f'Successfully saved {symbol} retail sentiment')
    return df


def balance_sheet_formatter(symbol:str):
    with open(f'data_equity/{symbol}_balance_sheet.dat','r') as file:
        f = file.read()
    j = json.loads(f)
    dat = j['quarterlyReports']
    df = pd.DataFrame(dat)
    df.rename(columns={'fiscalDateEnding': 'Date'},inplace=True)
    df.to_csv(f'csv_tests/{symbol}_balance_sheet.csv')


def earnings_init(symbol:str,start_date=None,end_date=None):
    with open(f'data_equity/{symbol}_earnings.dat','r') as file:
        r = file.read()
    data = pd.read_json(r,convert_axes=True)

    for index in data.index:
        if data.iloc[index,1] == 'None':
            data.drop(index=index,inplace=True)
        elif data.loc[index,'surprise']=='None':
            data.loc[index,'surprise'] = 0

    df = fn.data_fill(data,damping_columns=['surprise'],start_date=start_date,end_date=end_date)
    return df






#------------------------------------------------------------------------------------
#NEWS SENTIMENT
#------------------------------------------------------------------------------------
#for the marketwatch function
class IntervalDate:
    def __init__(self, current_date, time_interval):
        self.current_date = current_date
        self.time_interval = time_interval
        
    def _get_formatted_date(self, date):
        year = str(date.year)
        month = str(date.month).zfill(2)
        day = str(date.day).zfill(2)
        return year,month,day
    
    def get_current_date(self):
        return self._get_formatted_date(self.current_date)
    
    def get_next_date(self):
        next_date = self.current_date + timedelta(days=self.time_interval)
        return self._get_formatted_date(next_date)
    


def terminate_and_run_proton(path,terminate=True,run=True):
    protonvpn_path = path
    connect_command = [protonvpn_path]

    kill_command = 'taskkill /IM ProtonVPN.exe /F'
    kill_background = 'taskkill /F /IM ProtonVPN.WireGuardService.exe'
    kill_service = 'taskkill /F /IM ProtonVPNService.exe'

    if terminate:
        subprocess.run(kill_command)
        subprocess.run(kill_service)
        subprocess.run(kill_background)
        time.sleep(12)
        print('Successful termination')
    if run:
        # Start the VPN connection process
        vpn_process = subprocess.Popen(connect_command)
        time.sleep(20)  # Wait for the connection to establish
        print('Connected to VPN')

    # Optionally, keep the VPN connection process running
    #if vpn_process:
    #    vpn_process.wait()  # Wait for the process to complete



#not sure how to implement consistent start and end date functionality here 
def get_bloomberg_data(symbol,max_page:int=10,start_date=None,end_date=None,file_method='w'):
    df = {'Date':[],'Headline':[],'Summary':[]}
    for page in range(max_page):
        url = f"https://www.bloomberg.com/markets2/api/search?page={page}&query={symbol}&sort=time:desc"

        payload = {}
        headers = {
        'authority': 'www.bloomberg.com',
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'cookie': '_gcl_au=1.1.1258867486.1687890016; _pxvid=43f2ef4b-1517-11ee-9c98-4d76c2da6132; _rdt_uuid=1687890040318.60c499b1-45fd-4a4e-8f20-2f74cd36be33; professional-cookieConsent=new-relic|perimeterx-bot-detection|perimeterx-pixel|google-tag-manager|google-analytics|microsoft-advertising|eloqua|adwords|linkedin-insights; drift_aid=50f77712-c0e7-4dca-be00-1a3be2328a99; driftt_aid=50f77712-c0e7-4dca-be00-1a3be2328a99; _ga_NNP7N7T2TG=GS1.1.1690421511.1.1.1690421544.27.0.0; optimizelyEndUserId=oeu1691579311263r0.14078048370792384; _sp_v1_data=585912; _sp_su=false; _gid=GA1.2.899983733.1691579312; ccpaUUID=8ddf4e4d-c0f6-4360-a8b8-a8380fef7366; dnsDisplayed=true; ccpaApplies=true; signedLspa=false; bbgconsentstring=req1fun1pad1; _gcl_aw=GCL.1691579312.CjwKCAjw8symBhAqEiwAaTA__DO-uluShYNXs3DHq9qThKK2LjCpEYvmtIQC0s2nkstqwB9aO3_W5xoCYs8QAvD_BwE; _gcl_dc=GCL.1691579312.CjwKCAjw8symBhAqEiwAaTA__DO-uluShYNXs3DHq9qThKK2LjCpEYvmtIQC0s2nkstqwB9aO3_W5xoCYs8QAvD_BwE; bdfpc=004.7462060169.1691579312202; _reg-csrf=s%3AlyBG8tgXEI9gNgMwydkfiRDm.RlyuzWcn50CU7nh9OqEZ95DENiWW7I6ne4qIka7hhS8; pxcts=139f8f51-36a5-11ee-ab90-6964554c7146; _scid=150ea57a-2797-4865-a5c9-ae5f03bd0f1b; _fbp=fb.1.1691579312789.341111564; agent_id=5b10caf2-1327-404d-bdd9-a996d347e790; session_id=020044b7-d1ac-4592-878e-cec443ac6e02; session_key=79b794f0c70c696278e62a0cf85e93415508cbaa; gatehouse_id=9d7077e9-22af-491f-a236-042c63c2a8f0; geo_info=%7B%22countryCode%22%3A%22US%22%2C%22country%22%3A%22US%22%2C%22cityId%22%3A%225368361%22%2C%22provinceId%22%3A%225332921%22%2C%22field_p%22%3A%22E6A909%22%2C%22field_d%22%3A%22rr.com%22%2C%22field_mi%22%3A-1%2C%22field_n%22%3A%22hf%22%2C%22trackingRegion%22%3A%22US%22%2C%22cacheExpiredTime%22%3A1692184112798%2C%22region%22%3A%22US%22%2C%22fieldMI%22%3A-1%2C%22fieldN%22%3A%22hf%22%2C%22fieldD%22%3A%22rr.com%22%2C%22fieldP%22%3A%22E6A909%22%7D%7C1692184112798; _li_dcdm_c=.bloomberg.com; _lc2_fpi=b1166d620485--01h7czqv6gbrbhddx2qjmrg1mb; _gac_UA-11413116-1=1.1691579314.CjwKCAjw8symBhAqEiwAaTA__DO-uluShYNXs3DHq9qThKK2LjCpEYvmtIQC0s2nkstqwB9aO3_W5xoCYs8QAvD_BwE; _sctr=1%7C1691553600000; seen_uk=1; exp_pref=AMER; ln_or=eyI0MDM1OTMiOiJkIn0%3D; _cc_id=cb4452165a13807982d81de51e7402ec; panoramaId=7071f388f239d3ecb953ab1733d316d5393826fd6dfa307fa39b72341051eee0; panoramaIdType=panoIndiv; afUserId=044f5126-bd6b-48c9-b84d-1a6205bfd708-p; AF_SYNC=1691579323016; _sp_v1_p=192; _parsely_session={%22sid%22:4%2C%22surl%22:%22https://www.bloomberg.com/search?query=AAPL&page=3&sort=time:desc%22%2C%22sref%22:%22%22%2C%22sts%22:1691627981370%2C%22slts%22:1691625583270}; _parsely_visitor={%22id%22:%22pid=fb1a7e688db3d4e11dc092a353d07cce%22%2C%22session_count%22:4%2C%22last_session_ts%22:1691627981370}; _sp_v1_ss=1:H4sIAAAAAAAAAItWqo5RKimOUbLKK83J0YlRSkVil4AlqmtrlXQGVlk0kYw8EMOgNhaXkfSQGOiwGnzKYgF_3pWTZQIAAA%3D%3D; _sp_krux=true; euconsent-v2=CPwSZUAPwSZUAAGABCENDPCgAP_AAEPAABpYH9oB9CpGCTFDKGh4AKsAEAQXwBAEAOAAAAABAAAAABgQAIwCAEASAACAAAACGAAAIAIAAAAAEAAAAEAAQAAAAAFAAAAEAAAAIAAAAAAAAAAAAAAAAIEAAAAAAUAAEFAAgEAAABIAQAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgYtAOAAcAD8ARwA7gCBwEHAQgAiIBFgC6gGvAWUAvMBggDFgDwkBcABYAFQAMgAcABAADIAGgARAAjgBMACeAH4AQgAjgBSgDKgHcAd4A9gCTAEpAOIAuQBkgQAGARwAnYCmw0AEBcgYACApsRABAXIKgAgLkGQAQFyDoDQACwAKgAZAA4ACAAFwAMgAaABEACOAEwAJ4AXQAxAB-AFGAKUAZQA7wB7AEmAJSAcQA6gC5AGSDgAoAFwBHAEcAJ2ApshANAAWABkAFwATAAxACOAFKAMqAdwB3gEpAOoAsoBchAACARwlAMAAWABkADgARAAjgBMADEAI4AUYA7wDqAMkJAAgALgEcKQFAAFgAVAAyABwAEAANAAiABHACYAE8AMQAfgBRgClAGUAO8AlIB1AFyAMkKABAALgAyAIOAps.YAAAAAAAAAAA; consentUUID=87aa1011-a199-4670-9bfa-e8a090dc0452_22; country_code=US; _reg-csrf-token=qtPaStnF-joAKNZrqCPnkMRvrKeSRd3V9FQQ; _user-data=%7B%22status%22%3A%22anonymous%22%7D; _last-refresh=2023-8-10%201%3A2; _scid_r=150ea57a-2797-4865-a5c9-ae5f03bd0f1b; __sppvid=43fc3cd7-9fee-4242-ad89-772dd26f0988; _uetsid=13d2691036a511ee9072e745523f4a97; _uetvid=524378e0151711eeb02ba54b6db7a142; _px3=006a2178000e88b5a8383e394d20abe4dec20fe6b83fec951bb5462fd849eb60:FwQbebtd0GCPQQcjQh6Az/v7P3oymeKuiyEhxRdWi+jkaqslxMxMp6XHIFTbdnbVR18c+tpQWFgGzXZeME4ARw==:1000:6mHvOZ7bghHs9Vwo3KeoDBalw4QnYCNkRQ86Wex8kkb/kSAhQS7Ez3WTj+2sp8oIfTypK/JaqIz6uRQ0XEV+riKpp5m6hFvxfiXPlKY2/RUzrecx5lF5mVDetdpb2t+MFvJp7X7XZTaxejFyjHlzxShVPX0Rec9UipbH7A9OdQVnff7EbL7t+6oJc2BXDTPSHOLsISghRszRcWl9R/9eNQ==; _px2=eyJ1IjoiOTY0ZTJkODAtMzcxOS0xMWVlLWJlYTUtYmZhODc3YWIwYmY4IiwidiI6IjQzZjJlZjRiLTE1MTctMTFlZS05Yzk4LTRkNzZjMmRhNjEzMiIsInQiOjE2OTE2Mjk2NTM4NTQsImgiOiIxYWU0NTYwNWNhOWU0OWRlMGM3NmQyZWYyN2Y2ZTlmNjA0NzE3ZWQ2NDdmMGRiMTIwMGY3ZDAyYjFkOWQ2Y2IyIn0=; panoramaId_expiry=1692234153858; __gads=ID=76572d94ef31506c:T=1691579321:RT=1691629354:S=ALNI_MYlt1TkN2cOqMPx-hpXB8MARppYBQ; __gpi=UID=000009b26c3c0a8a:T=1691579321:RT=1691629354:S=ALNI_MbuI-GU0H0AoY52S8l18G7ro2rJQA; _ga=GA1.2.1454142836.1687890016; _pxde=fc60b9dadff19404c07f24b80fdb9b6d6a574085838e971e8f4b42cf3d6b5154:eyJ0aW1lc3RhbXAiOjE2OTE2Mjk0ODc5NjgsImZfa2IiOjAsImlwY19pZCI6W119; _gat_UA-11413116-1=1; _ga_GQ1PBLXZCT=GS1.1.1691627960.4.1.1691629514.56.0.0; exp_pref=AMER; country_code=US',
        'newrelic': 'eyJ2IjpbMCwxXSwiZCI6eyJ0eSI6IkJyb3dzZXIiLCJhYyI6IjE5ODI2OTciLCJhcCI6IjE0MDk1Mjc5OCIsImlkIjoiYjI3ODk2YWU0ZDNiNGRiYSIsInRyIjoiNGQ3MjY1MmZiMjRmNDUxNzNjZTIwOTY3ZTFkNjZkZDAiLCJ0aSI6MTY5MTYyOTUxNDUxNywidGsiOiIyNTMwMCJ9fQ==',
        'referer': f'https://www.bloomberg.com/search?query={symbol}&page={page}',
        'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'traceparent': '00-4d72652fb24f45173ce20967e1d66dd0-b27896ae4d3b4dba-01',
        'tracestate': '25300@nr=0-1-1982697-140952798-b27896ae4d3b4dba----1691629514517',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        }
        try:
            response = requests.request("GET", url, headers=headers, data=payload)
            j = response.json()
        except:
            time.sleep(5)
            continue
        try:
            for result in j["results"]:
                df['Date'].append(result["publishedAt"])
                df['Headline'].append(result["headline"])
                df['Summary'].append(result["summary"])
        except KeyError:
            continue

    df = pd.DataFrame(df)

    df.loc[:,'Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date',inplace=True)
    if start_date is not None:
        if start_date>df['Date'].iloc[0]:
            date = fn.find_closest_date(df,start_date,direction='left')
            df = df.loc[df['Date']>=date]
    if end_date is not None:
        if end_date<df['Date'].iloc[-1]:
            date = fn.find_closest_date(df,end_date,direction='right')
            df = df.loc[df['Date']<=date]

    data = df.to_json(orient='records',date_format='iso')
    if file_method is not None:
        with open(f'web_scraping/{symbol}_bloomberg.dat','w') as file:
            file.write(data)
    return df




#from requests_html import HTMLSession

def get_reuters_data(company,max_offset:int=50):
    df = {'Date':[],'Headline':[],'Summary':[]}

    for offset in range(max_offset):
        offset *= 20    
        #this was a pain in the ass to figure out
        url = f"https://www.reuters.com/pf/api/v3/content/fetch/articles-by-search-v2?query=%7B%22keyword%22%3A%22{company}%22%2C%22offset%22%3A{offset}%2C%22orderby%22%3A%22display_date%3Adesc%22%2C%22size%22%3A20%2C%22website%22%3A%22reuters%22%7D&d=151&_website=reuters"

        payload = {}
        headers = {
        'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
        'Referer': f'https://www.reuters.com/site-search/?query={company}&offset={offset}',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'sec-ch-ua-platform': '"Windows"',
        'Cookie': 'reuters-geo={"country":"US", "region":"-"}'
        }

        response = requests.request("GET", url, headers=headers, data=payload)
        j = response.json()

        try:
            for item in j["result"]["articles"]:
                date = item["published_time"]
                title = item["basic_headline"]
                description = item["description"]
                df['Date'].append(date)
                df['Headline'].append(title)
                df['Summary'].append(description)
        except: #in case one page of reuters data isn't formatted the same
            input("Change IP then hit enter to continue: ")
            continue   

    df = pd.DataFrame(df)
    data = df.to_json(orient='records',date_format='iso')
    with open(f'web_scraping/{company}_reuters.dat','w') as file:
        file.write(data)




#looks like I can only get data as far back as 2020... so I'm not going to use marketwatch. #time_interval is the gap between days that the api processes
def get_marketwatch_data(company,start_date:datetime=datetime(2016,1,1),end_date:datetime=datetime(2023,6,1),time_interval:int = 1, file_method='w'):
    data = {'Date':[],'Headline':[],'Summary':[]}
    indexer = 0
    #iterate through 2 days at a time
    current_date = start_date
    counter = 0
    #terminate_and_run_proton(r"C:\Program Files\Proton\VPN\v3.1.0\ProtonVPN.exe",terminate=False)
    #terminate_and_run_proton(r"D:\Program Files (x86)\Proton Technologies\ProtonVPN\ProtonVPN.exe",terminate=False)

    while current_date<=end_date:
        current_year,current_month,current_day = IntervalDate(current_date,time_interval).get_current_date()
        next_year,next_month,next_day = IntervalDate(current_date,time_interval).get_next_date()
        
        url = f"https://www.marketwatch.com/search?q={company}&ts=5&sd={current_month}%2F{current_day}%2F{current_year}&ed={next_month}%2F{next_day}%2F{next_year}&partial=true&tab=All%20News"

        payload = {}
        headers = {
        'authority': 'www.marketwatch.com',
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        #'cookie': 'optimizelyEndUserId=oeu1691638315857r0.7087681223582081; ccpaApplies=true; ccpaUUID=f1b52e2b-831e-4661-a601-26f45b9c22e3; ab_uuid=2bb6b0db-e60c-4a58-8c6f-1a7ffe71c69d; AMCVS_CB68E4BA55144CAA0A4C98A5%40AdobeOrg=1; _cls_v=71b4337b-4d2a-4857-8ab9-25e547a46069; _cls_s=cb9ce4cd-7044-42e8-b7e0-997bafce5259:0; _pcid=%7B%22browserId%22%3A%22ll4lrzf3xsz5befm%22%7D; cX_P=ll4lrzf3xsz5befm; _pctx=%7Bu%7DN4IgrgzgpgThIC4B2YA2qA05owMoBcBDfSREQpAeyRCwgEt8oBJAEzIE4AmHgZi4CsvAIwB2DqIAMADkHTRvEAF8gA; s_ecid=MCMID%7C07811295556865454993086478630926961519; s_cc=true; _uetvid=758743f0372e11eeaf7773af3d00e586; _rdt_uuid=1691638317494.509efc6d-b206-4dad-ab21-fa1245ae3898; _gcl_aw=GCL.1691638318.CjwKCAjw8symBhAqEiwAaTA__Ory9tIPdbQkDnXN5UtbxHrw9yWDTdXcRgrZFKSNTkdxN0A8UQViuBoCVrgQAvD_BwE; _gcl_au=1.1.1194856486.1691638318; cX_G=cx%3Aaay94xhyxbv11gicm2zvcbs06%3A118c0xph4rjn9; permutive-id=6c6c2bca-c05c-4138-88d6-6e1f360c4fc5; mw_loc=%7B%22Region%22%3A%22CA%22%2C%22Country%22%3A%22US%22%2C%22Continent%22%3A%22NA%22%2C%22ApplicablePrivacy%22%3A0%7D; fullcss-home=site-60d04d1451.min.css; icons-loaded=true; pushly.user_puuid=KjxevIlduSipjWShxOsKVchz9Kkg0XwA; letsGetMikey=enabled; dnsDisplayed=false; signedLspa=false; _pubcid=264ea218-4b67-4419-bbe9-75518d56c802; _ncg_domain_id_=1a8cf395-3657-4492-be2f-7e6cb4e543a6.1.1691638334184.1754710334184; _fbp=fb.1.1691638335144.1497534032; _dj_sp_id=e5a66968-eecf-41a1-a2f3-f6bb63e46721; _pcus=eyJ1c2VyU2VnbWVudHMiOm51bGx9; s_sq=djglobal%252Cdjwsj%3D%2526pid%253Dhttps%25253A%25252F%25252Fwww.marketwatch.com%25252F%2526oid%253D%25250A%252520%252520%252520%252520%252520%252520%2526oidt%253D3%2526ot%253DSUBMIT; fullcss-quote=quote-ccd11d2396.min.css; recentqsmkii=Stock-US-AAPL; _lr_env_src_ats=false; fullcss-section=section-15f53c310c.min.css; consentUUID=67c2a1aa-b753-49dc-a1af-16e72cc55240_22; _ncg_id_=1a8cf395-3657-4492-be2f-7e6cb4e543a6; AMCV_CB68E4BA55144CAA0A4C98A5%40AdobeOrg=1585540135%7CMCIDTS%7C19580%7CMCMID%7C07811295556865454993086478630926961519%7CMCAAMLH-1692328332%7C9%7CMCAAMB-1692328332%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1691730732s%7CNONE%7CMCAID%7CNONE%7CvVersion%7C4.4.0; _ncg_g_id_=4d40a8c1-d098-4817-9b73-adeb1ac41c21.3.1691665459.1754737942904; _lr_retry_request=true; kayla=g=786071f144dd473aafd9ff6b10f7a39f; gdprApplies=true; _dj_id.cff7=.1691638335.4.1691724198.1691723534.d51a0f87-c3c3-4e0d-aaec-cad543626ab5; usr_bkt=HY0f8Of9M1; _parsely_session={%22sid%22:4%2C%22surl%22:%22https://www.marketwatch.com/search?q=apple&ts=0&tab=All%2520News%22%2C%22sref%22:%22https://www.marketwatch.com/investing/stock/aapl%22%2C%22sts%22:1691726316810%2C%22slts%22:1691723534515}; _parsely_visitor={%22id%22:%22pid=bf2e4b8d-4749-421a-840f-2d21ae71034b%22%2C%22session_count%22:4%2C%22last_session_ts%22:1691726316810}; _lr_geo_location_state=ZH; _lr_geo_location=NL; _pnss=blocked; sso_fired_at=1691726326729; _pbjs_userid_consent_data=8871137552901317; __gads=ID=5f76ae834c3d2c87:T=1691638332:RT=1691726327:S=ALNI_MYeQa6qXb5p7Jn8m05Wegyo5l6AwQ; __gpi=UID=000009b28943ba2b:T=1691638332:RT=1691726327:S=ALNI_Ma3mi2vCUEEu9MQ552iyJ8f_OnyLg; _lr_sampling_rate=100; _ncg_sp_ses.f57d=*; cto_bundle=WExp8194MEJVYldSQ0NVaHFpalpDbTJFN1pETFUwdXR1RWdUMzZCYzluR1pMeVo5WEhmWUtva3ZkY2pQTjFWNlhpc2p6NEQwcmYlMkJqYlc0V29XSU05Q0MyUTRDV3dxQjEwa3NIYVpiJTJGTzNkZDVxYWhyc3lDYkpqcG1HZ2VCam5uSWNzaDlwYVEyZWRsaGVLa0RsVHlaU3BUcGtRJTNEJTNE; utag_main=v_id:0189dd803edd000cc3ddf5e2d12b0506f00e106700aee$_sn:5$_ss:0$_st:1691728136829$vapi_domain:marketwatch.com$_prevpage:MW_Search%3Bexp-1691729936835$ses_id:1691726329310%3Bexp-session$_pn:2%3Bexp-session; _ncg_sp_id.f57d=1a8cf395-3657-4492-be2f-7e6cb4e543a6.1691638334.5.1691726341.1691724198.3eeb2ae9-d043-4806-9840-a470e1a7ac38; ln_or=eyIzOTQyNDE3IjoiZCJ9; s_tp=5355; s_ppv=MW_Search%2C46%2C44%2C2461; gdprApplies=true',
        'newrelic': 'eyJ2IjpbMCwxXSwiZCI6eyJ0eSI6IkJyb3dzZXIiLCJhYyI6IjE2ODQyNzMiLCJhcCI6Ijc1NDg5OTM4MiIsImlkIjoiYjhmMjQwNDZjZWUxM2QxNiIsInRyIjoiMmFkMTRhZmI4YjZiNWJmNzczY2UyYzhlMjVjMzc3MDAiLCJ0aSI6MTY5MTcyNjgxNDc4NCwidGsiOiIxMDIyNjgxIn19',
        'referer': f'https://www.marketwatch.com/search?q={company}&ts=5&sd={current_month}%2F{current_day}%2F{current_year}&ed={next_month}%2F{next_day}%2F{next_year}&tab=All%20News',
        'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'traceparent': '00-2ad14afb8b6b5bf773ce2c8e25c37700-b8f24046cee13d16-01',
        'tracestate': '1022681@nr=0-1-1684273-754899382-b8f24046cee13d16----1691726814784',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        }

        #response = requests.request("GET", url, headers=headers, data=payload)
        '''
        if indexer%800 == 0 and indexer!=0: #i think I need this in order to stop the 403 request errors
            input("Change IP address then hit enter to continue: ")
        '''
        try:
            response = requests.request("GET", url, headers=headers, data=payload)
            response_text = response.text
            soup = BeautifulSoup(response_text, 'html.parser')

            # Find all elements with the class "article__headline"
            headline_elements = soup.find_all(class_="article__headline")
        except:
            time.sleep(5)
            continue

        if len(headline_elements) == 0:
            #terminate_and_run_proton(r"D:\Program Files (x86)\Proton Technologies\ProtonVPN\ProtonVPN.exe")
            terminate_and_run_proton(r"C:\Program Files\Proton\VPN\v3.1.0\ProtonVPN.exe")
            counter+=1
            continue
        else:
            counter=0

        if counter >= 4:
            return 0    

        # Iterate through headline elements and extract headlines and dates
        for headline_element in headline_elements:
            headline_text = headline_element.a.get_text(strip=True)
            summary_element = headline_element.find_next(class_="article__summary")
            # Find the corresponding timestamp element
            timestamp_element = headline_element.find_next(class_="article__timestamp")
            date_text = timestamp_element.get_text(strip=True)
            summary_text = summary_element.get_text(strip=True)
            
            if headline_text and date_text:
                data['Date'].append(date_text)
                data['Headline'].append(headline_text)
                data['Summary'].append(summary_text)    
        
        indexer+=1
        if indexer%20==0:
            print(indexer)
        current_date = current_date + timedelta(days=time_interval+1) #have to increment by 2 days because I'm getting two days worth of data at a time

    df = pd.DataFrame(data)
    j = df.to_json(orient='records',date_format='iso')

    if file_method is not None:
        with open(f'web_scraping/{company}_marketwatch.dat',f'{file_method}') as file:
            file.write(j)    
    return df



#please don't change your website format barrons
#pain. I get blocked before I can even get data before 2022... I'll have to use a try except
#even more pain... the dates only go back to 2022 for Barron's. I'll have to limit my usage to new stocks
def get_barrons_data(company,max_page:int=60,file_method='w'):
    
    df = {'Date':[],'Headline':[],'Summary':[]}
    for page in range(max_page):
        url = f'https://www.barrons.com/search?id=%7B%22query%22%3A%7B%22not%22%3A%5B%7B%22terms%22%3A%7B%22key%22%3A%22SectionType%22%2C%22value%22%3A%5B%22NewsPlus%22%5D%7D%7D%5D%2C%22and%22%3A%5B%7B%22terms%22%3A%7B%22key%22%3A%22languageCode%22%2C%22value%22%3A%5B%22en%22%2C%22en-us%22%5D%7D%7D%2C%7B%22date%22%3A%7B%22key%22%3A%22liveDate%22%2C%22value%22%3A%222023-08-10T19%3A59%3A59-04%3A00%22%2C%22operand%22%3A%22LessEquals%22%7D%7D%2C%7B%22date%22%3A%7B%22key%22%3A%22liveDate%22%2C%22value%22%3A%222022-08-10T00%3A00%3A00%2B00%3A00%22%2C%22operand%22%3A%22GreaterEquals%22%7D%7D%2C%7B%22terms%22%3A%7B%22key%22%3A%22Product%22%2C%22value%22%3A%5B%22Barrons.com%22%2C%22Barrons.com%20Automated%20Market%20Wraps%22%2C%22Barrons%20Blogs%22%2C%22Barrons%20Advisor%20Credits%20Video%22%2C%22Barrons%20Broadband%20Video%22%2C%22Barrons%20Summit%20Video%22%2C%22Barrons%20Video%20Live%20Q%26A%22%2C%22Barrons.com%20Webstory%22%2C%22Barrons%20Live%20Coverage%22%5D%7D%7D%5D%2C%22or%22%3A%5B%7B%22query_string%22%3A%7B%22value%22%3A%22apple%22%2C%22default_or_operator%22%3Atrue%2C%22parameters%22%3A%5B%7B%22property%22%3A%22headline%22%2C%22boost%22%3A3%7D%2C%7B%22property%22%3A%22keywords%22%2C%22boost%22%3A4%7D%2C%7B%22property%22%3A%22byline%22%2C%22boost%22%3A3%7D%2C%7B%22property%22%3A%22body%22%2C%22boost%22%3A4%7D%2C%7B%22property%22%3A%22section_name%22%2C%22boost%22%3A5%7D%5D%7D%7D%2C%7B%22full_text%22%3A%7B%22value%22%3A%22{company}%22%2C%22match_phrase%22%3Atrue%2C%22parameters%22%3A%5B%7B%22property%22%3A%22headline%22%2C%22boost%22%3A3%7D%2C%7B%22property%22%3A%22body%22%2C%22boost%22%3A4%7D%5D%7D%7D%5D%7D%2C%22sort%22%3A%5B%7B%22key%22%3A%22relevance%22%2C%22order%22%3A%22desc%22%7D%5D%2C%22count%22%3A20%7D%2Fpage%3D{page}&type=allesseh_search_full_v2'

        payload = {}
        headers = {
        'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
        'tracestate': '1022681@nr=0-1-1684273-1385912469-255f303fef1948f1----1691666216001',
        'traceparent': '00-6964ae46184ea5919bfb3c1175932e00-255f303fef1948f1-01',
        'sec-ch-ua-mobile': '?0',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'newrelic': 'eyJ2IjpbMCwxXSwiZCI6eyJ0eSI6IkJyb3dzZXIiLCJhYyI6IjE2ODQyNzMiLCJhcCI6IjEzODU5MTI0NjkiLCJpZCI6IjI1NWYzMDNmZWYxOTQ4ZjEiLCJ0ciI6IjY5NjRhZTQ2MTg0ZWE1OTE5YmZiM2MxMTc1OTMyZTAwIiwidGkiOjE2OTE2NjYyMTYwMDEsInRrIjoiMTAyMjY4MSJ9fQ==',
        'Referer': f'https://www.barrons.com/search?mod=DNH_S&query={company}&page={page+1}',
        'sec-ch-ua-platform': '"Windows"'
        }
        try:
            response = requests.request("GET", url, headers=headers, data=payload)
            data = response.json()
        except:
            input("Change IP address and press enter to continue: ")
            response = requests.request("GET", url, headers=headers, data=payload)
            data = response.json()
        ids = [item["id"] for item in data["collection"]]
        for id in ids:
            #print(id)
            if id.startswith("lc"):
                continue
            url = f"https://www.barrons.com/search?id={id}&type=article%7Ccapi"

            id_payload = {}
            id_headers = {
            'authority': 'www.barrons.com',
            'accept': '*/*',
            'accept-language': 'en-US,en;q=0.9',
            'cookie': 'wsjregion=na%2Cus; gdprApplies=false; ccpaApplies=true; ab_uuid=57746ccb-f0e2-4456-acf3-5bf962e1ecd9; usr_bkt=1p6sdUNJWD; _pcid=%7B%22browserId%22%3A%22ll52818318ms5m7h%22%7D; cX_P=ll52818318ms5m7h; _pctx=%7Bu%7DN4IgrgzgpgThIC4B2YA2qA05owMoBcBDfSREQpAeyRCwgEt8oBJAEzIE4AmHgZi4CsvAIwB2DqIAMADkHTRAkAF8gA; _lr_geo_location_state=CA; _lr_geo_location=US; _pbjs_userid_consent_data=6683316680106290; _pubcid=a9c66174-d226-4d66-96b3-b1de5cdb1c63; cX_G=cx%3Aaay94xhyxbv11gicm2zvcbs06%3A118c0xph4rjn9; ccpaUUID=c2d2144e-67e6-4ffb-a000-5fce7bc58548; AMCVS_CB68E4BA55144CAA0A4C98A5%40AdobeOrg=1; AMCV_CB68E4BA55144CAA0A4C98A5%40AdobeOrg=1585540135%7CMCIDTS%7C19580%7CMCMID%7C07811295556865454993086478630926961519%7CMCAAMLH-1692270742%7C9%7CMCAAMB-1692270742%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1691673142s%7CNONE%7CMCAID%7CNONE%7CvVersion%7C4.4.0; s_cc=true; _rdt_uuid=1691665942919.a6e1c4be-6736-4b59-bb15-f8bde9c1518d; _fbp=fb.1.1691665943068.1240242682; _cls_v=451db5d3-3a0a-4094-b5a0-40ddf833857a; _cls_s=f3b88220-d350-41e8-b5b9-c6a0cfbb0d99:0; _ncg_domain_id_=0ec0072f-4db3-49d7-ba37-b2560556712d.1.1691665942904.1754737942904; _parsely_session={%22sid%22:1%2C%22surl%22:%22https://www.barrons.com/%22%2C%22sref%22:%22https://www.google.com/%22%2C%22sts%22:1691665943811%2C%22slts%22:0}; _parsely_visitor={%22id%22:%22pid=5e443952-8844-4822-9928-a578fd3058a9%22%2C%22session_count%22:1%2C%22last_session_ts%22:1691665943811}; _ncg_g_id_=4d40a8c1-d098-4817-9b73-adeb1ac41c21.3.1691665459.1754737942904; __gads=ID=c0815736f6fedddf:T=1691665942:RT=1691665942:S=ALNI_MboodYppO-KeovmgXjnvWNX5fiBHw; __gpi=UID=000009b291b15d47:T=1691665942:RT=1691665942:S=ALNI_MaJtq0rnOlFeh7f7piV5qu5GnNPAA; _lr_retry_request=true; _lr_env_src_ats=false; _dj_sp_id=efedd005-ecfa-4efd-b74f-0f17486efade; s_sq=djglobal%3D%2526c.%2526a.%2526activitymap.%2526page%253DBOL_Home_US%252520Home%252520Page%252520Weekday%2526link%253DSearch%252520News%252520%252526%252520Quotes%2526region%253Droot%2526pageIDType%253D1%2526.activitymap%2526.a%2526.c; usr_prof_v2=eyJpYyI6NH0%3D; permutive-id=6c6c2bca-c05c-4138-88d6-6e1f360c4fc5; DJSESSION=country%3Dus%7C%7Ccontinent%3Dna%7C%7Cregion%3Dca; vcdpaApplies=false; regulationApplies=gdpr%3Afalse%2Ccpra%3Atrue%2Cvcdpa%3Afalse; optimizelyEndUserId=oeu1691666193209r0.5358536663461277; ResponsiveConditional_initialBreakpoint=lg; dnsDisplayed=false; signedLspa=false; spotim_visitId={%22visitId%22:%221904069a-ba0a-4552-b5b5-066e23e5e608%22%2C%22creationDate%22:%22Thu%20Aug%2010%202023%2007:16:36%20GMT-0400%20(Eastern%20Daylight%20Time)%22%2C%22duration%22:1}; cto_bundle=MQjLXl9EQURaY0RhRXRScURXR3N2QmQ3V1ltJTJGZzMlMkJrajcyajNPT2Fnak5uZVdXUUJDbFdTMDVCTVJDWW9DaHl4d2ZibjBZVmpJeHlIdW1tMUQ3U3FaekVIQUFPM0c0SFhsZHRPSjZLZ1ZVS0lmYWgwTTZvSSUyRlFVTDJTV1BCVWFaSTNTR0xkb0VhdTZvU0RhalBnJTJCWjBRM3FFQSUzRCUzRA; utag_main=v_id:0189df25c56a0089c6625ae32bf80506f001806700aee$_sn:1$_ss:0$_st:1691669030668$ses_id:1691665941867%3Bexp-session$_pn:10%3Bexp-session$_prevpage:BOL_ResearchTools_Barrons%20Search%3Bexp-1691670830674$vapi_domain:barrons.com; _dj_id.c19f=.1691665943.4.1691667231.1691666841.e009148b-6626-46e1-aca9-a0100440bd68..7d3f95ce-0d69-435d-ac55-ae276a1f1587.1691667231115.1; _ncg_sp_id.e48a=0ec0072f-4db3-49d7-ba37-b2560556712d.1691665943.4.1691667231.1691666841.d4208a56-bd96-43c4-a763-735896527758; _ncg_id_=0ec0072f-4db3-49d7-ba37-b2560556712d; _gcl_au=1.1.690478435.1691667231; s_tp=4377; s_ppv=BOL_ResearchTools_Barrons%2520Search%2C93%2C22%2C4081; sso_fired_at=1691667268066',
            'if-none-match': 'W/"3b20-ajufIk8tn/Y9ejFdBGj9pEolf9I"',
            'newrelic': 'eyJ2IjpbMCwxXSwiZCI6eyJ0eSI6IkJyb3dzZXIiLCJhYyI6IjE2ODQyNzMiLCJhcCI6IjEzODU5MTI0NjkiLCJpZCI6ImEyODQ0ZTlhNjMxNGEzMGUiLCJ0ciI6IjI4YWJlNjg4OGZkYjg1YmRiMzczMDE2NTU4Nzc4ZTAwIiwidGkiOjE2OTE2NjcyNjg0NTUsInRrIjoiMTAyMjY4MSJ9fQ==',
            'referer': f'https://www.barrons.com/search?mod=DNH_S&query={company}&page={page+1}',
            'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'traceparent': '00-28abe6888fdb85bdb373016558778e00-a2844e9a6314a30e-01',
            'tracestate': '1022681@nr=0-1-1684273-1385912469-a2844e9a6314a30e----1691667268455',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
            }

            try:
                id_response = requests.request("GET", url, headers=id_headers, data=id_payload)
                id_data = id_response.json()
            except:
                input("Change IP address and press enter to continue: ")
                id_response = requests.request("GET", url, headers=id_headers, data=id_payload)
                id_data = id_response.json()

            time_stamp = id_data["data"]["timestamp"]
            headline = id_data["data"]["headline"]
            #unfortunately, sometimes the articles don't have a summary
            try:
                summary = id_data["data"]["summary"]
            except KeyError:
                summary = ''
            timestamp_milliseconds = int(time_stamp)
            timestamp_seconds = timestamp_milliseconds / 1000

            # Convert timestamp to datetime object
            timestamp_date = datetime.fromtimestamp(timestamp_seconds)
            print(timestamp_date)
            df['Date'].append(timestamp_date)
            df['Headline'].append(headline)
            df['Summary'].append(summary)

    df = pd.DataFrame(df)
    j = df.to_json(orient='records',date_format='iso')

    if file_method is not None:
        with open(f'web_scraping/{company}_barrons.dat','w') as file:
            file.write(j)
    return df




def get_seekingalpha_analysis(symbol,start_date=datetime(2010,1,1),end_date=datetime(2023,6,1),time_interval=20,file_method='w'):
    data = {'Date':[],'Headline':[]}
    current_date = start_date
    next_date = current_date + timedelta(days=time_interval+1)

    if symbol=='BRK-B':
        symbol = 'BRK.B'

    terminate_and_run_proton(r"C:\Program Files\Proton\VPN\v3.1.0\ProtonVPN.exe",terminate=False)
    #terminate_and_run_proton(r"D:\Program Files (x86)\Proton Technologies\ProtonVPN\ProtonVPN.exe",terminate=False)
    counter=0
    while next_date <= end_date:
        #the url locates time through unix timestamps
        current_str = current_date.strftime('%Y-%m-%d')
        print(current_str)
        next_str = next_date.strftime('%Y-%m-%d')  
        since = str(current_date.timestamp())
        until = str((next_date-timedelta(seconds=1)).timestamp())
        compare_url = f'https://seekingalpha.com/api/v3/symbols/amd/analysis?filter[since]=1433131200&filter[until]=1435291199&id=amd&include=author%2CprimaryTickers%2CsecondaryTickers&isMounting=true&page[size]=40&page[number]=1'
        url = f"https://seekingalpha.com/api/v3/symbols/{symbol}/analysis?filter[since]={since}&filter[until]={until}&id={symbol}&include=author%2CprimaryTickers%2CsecondaryTickers&isMounting=true&page[size]=40&page[number]=1"

        payload = {}
        
        headers = {
        'authority': 'seekingalpha.com',
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        #'cookie': 'machine_cookie=9532473410670; LAST_VISITED_PAGE=%7B%22pathname%22%3A%22https%3A%2F%2Fseekingalpha.com%2Fsymbol%2FAMD%2Fanalysis%22%2C%22pageKey%22%3A%22843c2aae-9e56-4453-b6cd-7f71cc30bd05%22%7D; sailthru_pageviews=1; _gcl_au=1.1.1657915304.1692537895; _ga=GA1.1.967943935.1692537895; _uetsid=f3bd25003f5c11ee86d14b8124c4217c; _uetvid=f3bd61103f5c11eeb27b39c1af589ad6; _hjSessionUser_65666=eyJpZCI6IjgxOWEyMDFiLTUwOTYtNTAyOC1hNzc3LTQzZGY2MDJlNmNmYyIsImNyZWF0ZWQiOjE2OTI1Mzc4OTU1NDgsImV4aXN0aW5nIjpmYWxzZX0=; _hjFirstSeen=1; _hjIncludedInSessionSample_65666=0; _hjSession_65666=eyJpZCI6IjBmZjg1OTZiLTQ0ZjgtNDZkOC1hZTEyLTQwNDFkMWUzODNjOCIsImNyZWF0ZWQiOjE2OTI1Mzc4OTU1NTcsImluU2FtcGxlIjpmYWxzZX0=; _hjAbsoluteSessionInProgress=0; _hjHasCachedUserAttributes=true; sailthru_content=99a7dcdbb0310b96662f9129bbe1589a; sailthru_visitor=0e3e9f36-8463-4353-96ea-3e4cd2e29dad; _pcid=%7B%22browserId%22%3A%22lljhd2y2j8pr0u2v%22%7D; _pcus=eyJ1c2VyU2VnbWVudHMiOm51bGx9; __tbc=%7Bkpex%7DAW8lMjPLwvvS_mD7M9O3aPqDDiot6DjyTE_7VhJsVUAwA75AfFmA_eSFgY7p3f_X; __pat=-14400000; __pvi=eyJpZCI6InYtMjAyMy0wOC0yMC0wOS0yNC01Ni0yNDUtc1NYZGlTMExMemJZaFBjZi1kZWU5YjMzMTkzNzMzM2FiOWY1OTI5NTY5MzFjZDcxNCIsImRvbWFpbiI6Ii5zZWVraW5nYWxwaGEuY29tIiwidGltZSI6MTY5MjUzNzg5NjQ0MH0%3D; _pctx=%7Bu%7DN4IgrgzgpgThIC4B2YA2qA05owMoBcBDfSREQpAeyRCwgEt8oBJAE0RXQF8g; xbc=%7Bkpex%7DZtsgYmRp4jU5IX7fsCQMtsGOTC5M93XQv5RXQJvfAHUqs0Pm1IZMg4VztNgGEIJxLb5Uo2tx5mTd_mb2zVwonisUz6UbllyhI5XoiU7TqlV_3M2SUdCGiwoW5UxJ6THxz0u0RiSopyTW5iaT9Ke8ND8bUt-ecF89dwMQ98fqlSQHkiPlKAHN6apBexP9CKRw; pxcts=f35bdcab-3f5c-11ee-ae5e-79646442466b; _pxvid=f35bcd7d-3f5c-11ee-ae5e-6119e0bbd011; _px2=eyJ1IjoiZjQ0MjQ5MzAtM2Y1Yy0xMWVlLWFlYjAtYjk5MWM4NTk3Y2M3IiwidiI6ImYzNWJjZDdkLTNmNWMtMTFlZS1hZTVlLTYxMTllMGJiZDAxMSIsInQiOjE2OTI1MzgzOTU2MjEsImgiOiI4M2NkMTA1MDIxYWVkMTRhM2NmMmE0OTZmMTY3ODQwZTAxOTE1ZGM0MDA0ZTlhODczOWM5MGE1OWZjYzcyMDI4In0=; _px=dsx6kPE+38ayh610k86DHMmaUAy4bZe/jyLY3k9BL1vwhUmJxVa7S57qpFtaeUmxq9TsBLgYf0njDcFCioBz0w==:1000:KCUKlo7F6BjUJCc07KEZD4vxGXOuToCdF30tLuXOXtaMUnwvqznrbJMTvSndFgXP6QpyHwd2yko4S0wVDJQDtrZt6EIAWtmYYuEhwBk6xZSjK0YEGH3wEUGbMSGVIKigkqq7V4dgD2meexs5/XHFvlU+n5R/O4aiTSVt29jpiRA6gA0ACdbjkmiMqlEicsnqBAVZXICTj8wKACThmlKW9d9VsjKigtlByjOu3+wincsU9bEkg+CLUtBFEu4mjqFj0+UTgScTnr0EXabY7lQnWQ==; _pxde=c09a5289f30de66e5915a535f988e844e25c53a2911abc95839afa0f12604379:eyJ0aW1lc3RhbXAiOjE2OTI1Mzc5MTE3OTgsImZfa2IiOjB9; _ga_KGRFF2R2C5=GS1.1.1692537895.1.1.1692537926.29.0.0',
        'referer': f'https://seekingalpha.com/symbol/{symbol}/analysis?from={current_str}T04%3A00%3A00.000Z&to={next_str}T03%3A59%3A59.999Z',
        'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        }
        try:
            response = requests.request("GET",url,headers=headers, data=payload)
            j = response.json()
        except:
            time.sleep(15)
            continue
        if not 'data' in j:
            #terminate_and_run_proton(r"D:\Program Files (x86)\Proton Technologies\ProtonVPN\ProtonVPN.exe")
            terminate_and_run_proton(r"C:\Program Files\Proton\VPN\v3.1.0\ProtonVPN.exe")
            response = requests.request("GET", url, headers=headers, data=payload)
            j = response.json()
            counter+=1
            if not 'data' in j:
                print('No data')
                current_date = next_date
                next_date = next_date + timedelta(days=time_interval+1)
                continue
        else:
            counter=0
        
        if counter>=4:
            return 0

        for article in j['data']:
            attributes = article.get('attributes', {})
            title = attributes.get('title', None)
            publish_on = attributes.get('publishOn', None)
            
            if title is not None and publish_on is not None:
                data['Date'].append(publish_on)
                data['Headline'].append(title)

        current_date = next_date
        next_date = next_date + timedelta(days=time_interval+1)

    data['Date'] = data['Date'].apply(lambda x: x[:10])
    df = pd.DataFrame(data)
    j = df.to_json(orient='records',date_format='iso')

    if symbol == 'BRK.B':
        symbol = 'BRK-B'

    if file_method is not None:
        with open(f'web_scraping/{symbol}_seekingalpha.dat',file_method) as file:
            file.write(j)
    return df



def get_cnbc_data(company, scroll: int = 30,file_method='w'):
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run Chrome in headless mode (no UI)
    driver = webdriver.Chrome(options=options)
    
    url = f'https://www.cnbc.com/search/?query={company}&qsearchterm={company}'
    driver.get(url)

    df = {'Date': [], 'Headline': [], 'Summary': []}
    
    # Scroll down the page to load more content
    for _ in range(scroll):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3)
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    search_results = soup.find_all(class_='SearchResult-searchResult')
    
    for result in search_results:
        title_element = result.find(class_='Card-title')
        
        if title_element:
            title_text = title_element.get_text()
            
            author_element = result.find(class_='SearchResult-author')
            date_element = result.find(class_='SearchResult-publishedDate')
            preview_element = result.find(class_='SearchResult-searchResultPreview')
            
            author = author_element.get_text() if author_element else "N/A"
            date = date_element.get_text() if date_element else "N/A"
            preview = preview_element.get_text() if preview_element else "N/A"

            df['Date'].append(date)
            df['Headline'].append(title_text)
            df['Summary'].append(preview)
    
    df = pd.DataFrame(df)
    j = df.to_json(orient='records',date_format='iso')

    if file_method is not None:
        with open(f'web_scraping/{company}_cnbc.dat','w') as file:
            file.write(j)

    driver.quit()
    return df


#------------------------------------------------------------------------------
#HUGGING FACE PRE TRAINED SENTIMENT MODEL
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from scipy.special import softmax

#this model is specifically trained for financial news sentiment.
MODEL = f"mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def roberta_sentiment(sentence:str):
    encoded_text = tokenizer(sentence,return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy() #returns sentiment scores as a np array of the form [neg,neutral,pos]
    scores = softmax(scores)
    #lets return one number in the range (-1,1), -1 being as negative as possible and 1 being as positive as possible
    sentiment = scores[0]*(-1.) + scores[1]*0. + scores[2]*1.
    return sentiment


def news_roberta(company,outlet,start_date,end_date): #pretty much the same as bloomberg init
    with open(f'web_scraping/{company}_{outlet}.dat','r') as file:
        r = file.read()
    data = pd.read_json(r)

    if outlet == 'seekingalpha':
        data['Date'] = data['Date'].apply(lambda x: x[:10])
    
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by='Date', axis=0)
    
    #might need some conditionals in here to make sure start_date>min_date and likewise end_date<max_date
    if start_date is not None:
        date = fn.find_closest_date(data,start_date,direction='left')
        data = data.loc[data['Date']>=date]
    if end_date is not None:
        date = fn.find_closest_date(data,end_date,direction='right')
        data = data.loc[data['Date']<=end_date]
    data.reset_index(inplace=True,drop=True)

    sentiment_df = {'Headline_score':[]}

    for index in data.index:
        headline_score = roberta_sentiment(data['Headline'].iloc[index])
        #summary_score = roberta_sentiment(data['Summary'].iloc[index])
        sentiment_df['Headline_score'].append(headline_score)
        #sentiment_df['Summary_score'].append(summary_score)
    sentiment_df = pd.DataFrame(sentiment_df)
    #print(sentiment_df)
    df = pd.concat([data,sentiment_df],axis=1)
    #print(df)
    df = df[['Date','Headline_score']]
    #df.reset_index(inplace=True)
    return df


#the news_init function takes a while so I'm going to initialize one at a time and save them to dat files.
def news_formatter(symbol,
                   outlets=['bloomberg','marketwatch'],
                   start_date=pd.Timestamp('2010-01-01'), 
                   end_date=pd.Timestamp('2023-06-01'),
                   file_method = 'w',
                   file_prefix='news_sentiment'):
    df_list = []
    for outlet in outlets:
        df_list.append(news_roberta(symbol,outlet,start_date=start_date,end_date=end_date))

    if len(df_list)>1:
        '''
        new_list = []
        for df in df_list:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date')
            df = df[df['Date'] >= start_date] #filter out unwanted dates
            df = df[df['Date'] <= end_date]
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            df['Date'] = pd.to_datetime(df['Date'])
            #df.drop(columns=['Summary'],inplace=True) #This is already done in news_roberta
            df.reset_index(inplace=True,drop=True)
            new_list.append(df)
        '''
        df = pd.concat(df_list,axis=0)
    else:
        df = df_list[0]

    #print(df)
    #collapse all the headline scores into one averaged column
    #average_series = df.apply(lambda row: row[1:].mean(), axis=1)
    #print(average_series)
    # Convert the average Series back to a DataFrame
    #average_df = pd.concat([df.iloc[:,0],pd.DataFrame({'Headline': average_series})],axis=1)

    df = fn.data_fill(df,interpolate_columns='all',start_date=start_date,end_date=end_date)
    j = df.to_json(orient='records',date_format='iso')
    if file_method is not None:
        with open(f'data_misc/{file_prefix}_{symbol}.dat',file_method) as file:
            file.write(j)
    print(f'Successfully formatted {symbol} news data')
    return df



def news_init(symbol,prefix='news_sentiment',start_date=None,end_date=None):
    with open(f'data_misc/{prefix}_{symbol}.dat','r') as file:
        r = file.read()
    df = pd.read_json(r)
    if end_date is not None:
        df = fn.data_fill(df,end_date=end_date)
    return df







#--------------------------------------------------------------
#TREND DATA
#--------------------------------------------------------------
from pytrends.request import TrendReq
pytrends = TrendReq(hl='en-US',tz=360)

def form_two_columns(row):
    row_str = str(row)
    row_list = row_str.split('\n',2)
    value = float(row_list[0].split(' ',1)[1])
    date = (row_list[2].split(' ',2)[1])
    return [date,value]

def get_search_trend_data(company,start_date=datetime(2004,6,1),end_date=datetime(2023,6,1),file_method='w'):
    kw_list = [company]
    current_date = start_date
    dataframe = {'Date':[],'Value':[]}

    while current_date<=end_date:
        inc_date = current_date+timedelta(days=90)
        if inc_date>end_date:
            inc_date = end_date

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

    if file_method is not None:
        trend_json = df.to_json(orient='records',date_format='iso')
        with open(f'data_misc/trend_{company}.dat',file_method) as file:
            file.write(trend_json)
    return df


def trend_init(symbol:str,start_date=None,end_date=None):
    with open(f'data_misc/trend_{symbol}.dat', 'r') as file:
        data = file.read()
    trend_df = pd.read_json(data,convert_dates=True)
    trend_df = fn.data_fill(trend_df,start_date=start_date,end_date=end_date)
    return trend_df








#--------------------------------------------------------------------
#FED DATA & HOUSING DATA
#--------------------------------------------------------------------
fed_params = {'GDP':(False,True),'CPIAUCSL':(False,True),
            'DFF':(True,False),'M1':(False,True),'M1V':(True,True),
            'INDPRO':(False,True),'UNRATE':(True,False)}


#see https://data.nasdaq.com/data/FRED-federal-reserve-economic-data/documentation
def get_fed_data(code,start_date=datetime(1990,1,1),end_date=datetime(2023,6,1),file_method='w'):
    start_date = datetime.strftime(start_date,'%Y-%m-%d')
    end_date = datetime.strftime(end_date,'%Y-%m-%d')
    
    code_get = quandl.get(f'FRED/{code}',start_date=start_date,end_date=end_date)
    code_df = pd.DataFrame(code_get)
    new_df = {'Date':[],'Value':[]}

    for index,value_row in code_df.iterrows():
        arr = form_two_columns(value_row)
        new_df['Date'].append(arr[0])
        new_df['Value'].append(arr[1])

    code_df = pd.DataFrame(new_df)
    code_df['Date'] = pd.to_datetime(code_df['Date'])
    code_json = code_df.to_json(orient='records',date_format='iso')

    if file_method is not None:
        with open(f'data_fed/{code}.dat','w') as file:
            file.write(code_json)
    print(f'Got {code} FED data')
    return code_df


def fed_formatter(code,start_date,end_date,nominal:bool=True,interpolate:bool=True):
    with open(f'data_fed/{code}.dat','r') as file:
        d = file.read()
    df = pd.read_json(d)
    df.loc[:,'Date'] = pd.to_datetime(df['Date'])
    if start_date is not None:
        df = df.loc[df['Date']>=start_date]
    else:
        df = df.loc[df['Date']>=pd.to_datetime('1990-01-01')]

    columns_list = list(df.columns[1:])
    if nominal==True:
        df_fill = fn.data_fill(df,interpolate_columns=columns_list)
    else:
        columns_list = list(df.columns[1:])
        df_fill = fn.data_fill(df,percent_columns=columns_list,interpolate_columns=columns_list,start_date=start_date,end_date=end_date)
    print(f'Successfully formatted {code} data')
    return df_fill


def fed_init(code_list,start_date=None,end_date=None):
    fed_data = {}

    for code in code_list:
        fed_data[code] = fed_formatter(code,start_date,end_date,nominal=fed_params[code][0],interpolate=fed_params[code][1])

    print("Initialized fed data")
    return fed_data


def fed_eval_init(df,code):
    columns_list = list(df.columns[1:])
    nominal = fed_params[code][0]
    interpolate = fed_params[code][1]

    if nominal==True:
        df_fill = fn.data_fill(df,interpolate_columns=columns_list)
    else:
        columns_list = list(df.columns[1:])
        df_fill = fn.data_fill(df,percent_columns=columns_list,interpolate_columns=columns_list)
    return df_fill



#-------------------------------------------------------------------------
city_ids = {'NY':394913,'LA':753899,'Chicago':394463}


def get_housing_data(city,start_date=datetime(2000,1,1),end_date=datetime(2023,6,1),file_method='w'):
    #start_date = datetime.strftime(start_date,'%Y-%m-%d')
    #end_date = datetime.strftime(end_date,'%Y-%m-%d')

    city_id = city_ids[city]
    
    metro = quandl.get_table('ZILLOW/DATA', indicator_id='ZSFH', region_id=city_id)
    df = pd.DataFrame(metro)
    df.rename(columns={'date':'Date','value':city},inplace=True)
    df.loc[:,'Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date')
    df = df.loc[df['Date']>=start_date]
    df = df.loc[df['Date']<=end_date]

    j = df.to_json(orient='records',date_format='iso')
    j.strip('[]')

    if file_method is not None:
        with open(f'data_misc/housing_{city}.dat',file_method) as file:
            file.write(j)
    return df


def housing_formatter(city:str,start_date,end_date):
    with open(f'data_misc/housing_{city}.dat','r') as f:
        f_read = f.read()
    df = pd.read_json(f_read)
    df = df[['Date',city]]
    output_df = fn.data_fill(df,percent_columns=[city],interpolate_columns=[city],start_date=start_date,end_date=end_date)
    return output_df


def housing_init(city_list,start_date=None,end_date=None):
    housing_data = {}
    for city in city_list:
        housing_data[city] = housing_formatter(city,start_date=start_date,end_date=end_date)
    print("Initialized housing data")
    return housing_data


def housing_eval_init(df,city,start_date):
    df = df[['Date',city]]
    output_df = fn.data_fill(df,percent_columns=[city],interpolate_columns=[city],start_date=start_date)
    return output_df    






#-----------------------------------------------------------------
#SEC DATA
#-----------------------------------------------------------------
#IMPORTANT: the sec locates company urls by the "CIK" number, which is a unique identifier for each company. I'm going to have to include them in a dictionary.
cik = {'AAL':6201,'AAPL':320193,'AMD':2488,
       'AMZN':1018724,'BAC':70858,'BANC':1169770,'BRK-B':1067983,'CGNX':851205,
       'CSCO':858877,'DELL':1571996,'DIS':1744489,'F':37996,'GE':40545,
       'GOOG':1652044,'INTC':50863,'MCD':63908,'META':1326801,'MLM':916076,'MSFT':789019,'NFLX':1065280,
       'NVDA':1045810,'QCOM':804328,'ROKU':1428439,'RUN':1469367,'SBUX':829224,'SHOP':1594805,
       'T':732717,'TGT':27419,'TSLA':1318605,'UPS':1090727,'WMT':104169}


def get_insider_trading_data(symbol:str,max_page:int=6,start_date=datetime(2005,1,1),end_date=datetime(2023,6,1),file_method='w'):
    #I'm thinking about doing percentage of securities owned vs acquired or sold
    pages = []
    for n in range(max_page):
        pages.append(80*n) #the sec includes 80 rows of data per page

    df = {'Date':[],'InsiderFlow':[]} #change will be negative for a sell, positive for a purchase
    locator = str(cik[symbol]).zfill(10)

    for num in pages:
        url = f'https://www.sec.gov/cgi-bin/own-disp?action=getissuer&CIK={locator}&type=&dateb=&owner=include&start={num}'
        response = requests.get(url)
        html_content = response.content

        soup = BeautifulSoup(html_content, "html.parser")
        table = soup.find("table", {"id": "transaction-report"})

        for row in table.select('tr:not(.header)'):
            data = row.find_all("td")

            buy_or_sell = data[0].text.strip()
            date = data[1].text.strip()

            code = data[5].text.strip()
            if not (code.startswith('P') 
                    or code.startswith('S') 
                    or code.startswith('A') 
                    or code.startswith('F')):
                continue

            try:
                first_number = float(data[7].text.strip())
                second_number = float(data[8].text.strip())
            except:
                continue
            if second_number == 0:
                continue #I'll just skip this iteration if the shares owned is zero because I'm not sure what to replace it with

            change = first_number/second_number
            if buy_or_sell == 'D':
                change *= -1
            df['Date'].append(date)
            df['InsiderFlow'].append(change)
    
    df = pd.DataFrame(df)
    df.loc[:,'Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date')

    #start_date = pd.to_datetime(datetime.strftime(start_date,'%Y-%m-%d'))
    #end_date = pd.to_datetime(datetime.strftime(end_date,'%Y-%m-%d'))
    df = df.loc[df['Date']>=start_date]
    df = df.loc[df['Date']<=end_date]

    j = df.to_json(orient='records',date_format='iso')

    if file_method is not None:
        with open(f'data_misc/insider_{symbol}.dat',file_method) as file:
            file.write(j)
    return df


def insider_init(symbol:str,start_date=None,end_date=None):
    with open(f'data_misc/insider_{symbol}.dat','r') as file:
        r = file.read()
    data = pd.read_json(r,convert_axes=True)
    #replace all rows where insider_flow==1 to =0
    data['InsiderFlow'] = data['InsiderFlow'].replace(1,0)
    df = fn.data_fill(data,start_date=start_date,end_date=end_date)
    return df











#----------------------------------------------------------------------------------
#MASTER INITIALIZERS
#----------------------------------------------------------------------------------

#IMPORTANT: since I want to look at how the present day data effects the closing price of tomorrow, I have to copy the close column, and shift it up 1 position in a new row
def training_data_init(stocks_list,
                   city_list = [],
                   fed_list = [],
                   news_prefixes = ['news_sentiment','seekingalpha'], #this will typically just be [news_sentiment,seekingalpha]
                   rsi = True, #include RSI momentum indicator
                   earnings = True,
                   retail_sentiment=True,
                   google_trend=True,
                   insider_data=True,
                   start_date=None,
                   end_date=None,
                   file_name='TRAINING'):
    
    df_list = []
    if city_list:
        housing_data = housing_init(city_list,start_date=start_date,end_date=end_date)
    if fed_list:
        fed_data = fed_init(fed_list,start_date=start_date,end_date=end_date)
        
    for i in range(len(stocks_list)):
        data_list = []
        sym = stocks_list[i]

        stock = fn.equity_formatter(sym)
        data_list.append(stock)

        if earnings:
            stock_earnings = earnings_init(sym,end_date=end_date)
            data_list.append(stock_earnings)

        if google_trend:
            trend = trend_init(sym,start_date=start_date,end_date=end_date)
            data_list.append(trend)

        if news_prefixes:
            for prefix in news_prefixes: 
                data_list.append(news_init(sym,prefix=prefix,start_date=start_date,end_date=end_date))

        if rsi:
            stock_rsi = fn.calculate_rsi(sym)
            data_list.append(stock_rsi)

        if insider_data:
            insider = insider_init(sym,start_date=start_date,end_date=end_date)
            data_list.append(insider)

        if retail_sentiment:
            retail_df = fn.retail_sentiment_formatter(sym,start_date=start_date,end_date=end_date)
            data_list.append(retail_df)
        
        for code in fed_list:
            data_list.append(fed_data[code])

        for city in city_list:
            data_list.append(housing_data[city])

        if len(data_list)<2:
            df=data_list[0]
        else:
            df = fn.concatenate_data(data_list,start_date=start_date,end_date=end_date)
            df.to_csv(f'csv_tests/concatenate_{sym}.csv')

        df['Close_Tmr'] = df['Close'].shift(-1)
        df = df.drop(index=df.index[-1]) #drop the last row because the above shift method will result in NaN values
        df = df.fillna(0)
        df.set_index('Date',inplace=True)
        print(f'Finished formatting {sym}')
        df.to_csv(f'csv_tests/formatted_{sym}.csv')
        df_list.append(df)

    if len(df_list)<2:
        data = df_list[0]
    else:
        data = pd.concat(df_list,axis=0,ignore_index=True)
    data.to_csv(f'csv_data/{file_name}.csv')
    print(f'Initialized sentiment data')
    return data



def eval_data_init(stock:str, #stock symbol
                   city_list = [],
                   fed_list = [],
                   news_outlets = ['bloomberg','marketwatch'],
                   rsi = True, #include RSI momentum indicator
                   earnings = True,
                   retail_sentiment=True,
                   google_trend=True,
                   insider_data=True,
                   hist=256): #hist is the number of days of history you want to include in the evaluation tensor
    end_date = datetime.now()
    start_date = end_date - timedelta(days=hist)

    df_list = []

    #equity price
    equity_pth = f'data_equity/{stock}.dat'
    equity_df = data_appender(equity_pth,get_equity_data,stock,start_date=start_date,end_date=end_date)
    equity_df = fn.equity_formatter(stock,start_date=start_date,end_date=end_date)
    df_list.append(equity_df)

    #earnings
    if earnings:
        earnings_pth = f'data_equity/{stock}_earnings.dat'
        earnings_df = data_appender(earnings_pth,get_earnings,stock,start_date=start_date,end_date=end_date)
        earnings_df = earnings_init(stock,start_date=start_date,end_date=end_date)
        df_list.append(earnings_df)
        print('earnings')
        print(earnings_df)
    
    if google_trend:
        trend_pth = f'data_misc/trend_{stock}.dat'
        trend_df = data_appender(trend_pth,get_search_trend_data,stock,start_date=start_date,end_date=end_date)
        trend_df = trend_init(stock,start_date=start_date,end_date=end_date)
        df_list.append(trend_df)
        print('trend')
        print(trend_df)

    #news sentiment
    def news_pth(outlet):
        pth = f'web_scraping/{stock}_{outlet}.dat'
        return pth

    if news_outlets:
        for outlet in news_outlets:
            pth = news_pth(outlet)
            if outlet == 'seekingalpha':
                data_appender(pth,get_seekingalpha_analysis,stock,start_date=start_date,end_date=end_date)
            elif outlet == 'bloomberg':
                data_appender(pth,get_bloomberg_data,stock,start_date=start_date,end_date=end_date)
            elif outlet == 'marketwatch':
                data_appender(pth,get_marketwatch_data,stock,start_date=start_date,end_date=end_date)

        news_sentiment_df = news_formatter(stock,start_date=start_date,end_date=end_date,file_method=None)
        df_list.append(news_sentiment_df)
        print(news_sentiment_df)

        #alpha_df = news_formatter(stock,outlets=['seekingalpha'],start_date=start_date,end_date=end_date,file_method=None)
        #df_list.append(alpha_df)
        #print(alpha_df)    

    if rsi:
        stock_rsi = fn.calculate_rsi(stock,start_date=start_date,end_date=end_date)
        df_list.append(stock_rsi)
        print(stock_rsi)

    if insider_data:
        indsider_pth = f'data_misc/insider_{stock}.dat'
        insider_df = data_appender(indsider_pth,get_insider_trading_data,stock,start_date=start_date,end_date=end_date)
        insider_df = insider_init(stock,start_date=start_date,end_date=end_date)
        df_list.append(insider_df)
        print(insider_df)
    
    if retail_sentiment:
        retail_pth = f'data_equity/{stock}_retail_sentiment.dat'
        retail_df = data_appender(retail_pth,get_retail_sentiment,stock,start_date=start_date,end_date=end_date)
        retail_df = fn.retail_sentiment_formatter(stock,start_date=start_date,end_date=end_date)
        df_list.append(retail_df)
        print(retail_df)
    
    #federal reserve:
    def fed_pth(code):
        pth = f'data_fed/{code}.dat'
        return pth
    
    if fed_list:
        for code in fed_list:
            pth = fed_pth(code)
            fed_df = data_appender(pth,get_fed_data,code,start_date=start_date,end_date=end_date)
            fed_df = fed_eval_init(fed_df,code)
            df_list.append(fed_df)
            print(fed_df)
        print('Initialized fed data')

    #housing:
    def housing_pth(city):
        pth = f'data_misc/housing_{city}.dat'
        return pth
    
    if city_list:
        for city in city_list:
            pth = housing_pth(city)
            city_df = data_appender(pth,get_housing_data,city,start_date=start_date,end_date=end_date,overwrite=False)
            city_df = housing_formatter(city,start_date=start_date,end_date=end_date)
            df_list.append(city_df)
            print(city_df)
        print('Initialized housing data')

    df_merged = df_list[0]
    for df in df_list[1:]:
        df_merged = pd.merge(df_merged,df,on='Date',how='outer')

    date_str = datetime.strftime(end_date,'%Y-%m-%d')
    df_merged.to_csv(f'csv_data/{stock}-{date_str}')
    print(df_merged)
    return df_merged





#----------------------------------------------------------------------------
#EXECUTABLE CODE
#----------------------------------------------------------------------------

#sentiment_init(retail_stocks,retail_companies)
#equity_init(model_stocks,model_companies)
#as of now, seekingalpha_F.dat ends in 2015, so I am leaving ford out of the list
model_stocks = ['AAL','AAPL','AMD','AMZN','BAC','BANC','BRK-B','CGNX','UPS','DELL','DIS',
                'INTC','GE','TGT','MCD','MSFT','NVDA','NFLX','QCOM','ROKU','RUN','SBUX','WMT','GOOG','CSCO']
#sentiment_init(model_stocks)
pth = f'data_equity/{model_stocks[0]}_retail_sentiment.dat'
#data_appender(pth,get_retail_sentiment,model_stocks[0])

'''
training_data_init(['AAL'],
                   city_list=['NY'],
                   fed_list=['DFF','UNRATE'],
                   end_date=datetime(2023,6,1),
                   retail_sentiment=False,
                   file_name='AAL_TRAINING'
                   )
'''

eval_data_init(
    'AAPL',
    city_list=['NY'],
    #fed_list=['DFF','UNRATE']

)


#get_bloomberg_data('AAPL',max_page=40)