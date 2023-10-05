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
import os
import subprocess

ndaq = pd.read_csv("csv_data/nasdaq_screener.csv")
ndaq = ndaq.sort_values(by='Volume',ascending=False)

quandl.ApiConfig.api_key = 'h3gFTxuuELMsc6zXU7_J'

alphavantage_key='O7TXW0XZOPYNKC5D'
#alphavantage_key = '1C3O71BAB7HJXTWZ'
#alphavantage_key = 'B71NVO0WITDQFLF4'
benzinga_key = '4edd94fa08d140a78309628f689b7ada'

proton_path = r"C:\Program Files\Proton\VPN\v3.1.1\ProtonVPN.exe"
#proton_path = r"D:\Program Files (x86)\Proton Technologies\ProtonVPN\ProtonVPN.exe"

def change_alphavantage_key(key):
    if key=='1C3O71BAB7HJXTWZ':
        new_key = 'O7TXW0XZOPYNKC5D'
    elif key=='B71NVO0WITDQFLF4':
        new_key = '1C3O71BAB7HJXTWZ'
    else:
        print('error')
    return new_key



def data_appender(pth: str,
                  get_function: callable,
                  symbol: str,
                  start_date: datetime = None,
                  end_date: datetime = datetime.now(),
                  concat: bool = True,
                  sequence_length = 5,
                  overwrite=True):
    # Create the full file path by joining the cwd and the provided path
    full_path = os.path.join(os.getcwd(), pth)

    if overwrite==True and concat==False:
        print('Caution: recommended setting is concat=True when overwrite=True')

    if start_date is None:
        start_date = end_date - timedelta(days=sequence_length*3)

    if os.path.isfile(full_path) and os.path.getsize(full_path) > 0:
        with open(full_path, 'r') as f:
            df = pd.read_json(f, convert_dates=True)

        #df.loc[:,'Date'] = pd.to_datetime(df['Date'])

        recent_date = df['Date'].iloc[-1]

        if recent_date + timedelta(days=1) > end_date:
            print('No new data to append')
            start_date = fn.find_closest_date(df, start_date, direction='left')
            df = df.loc[df['Date'] >= start_date]
            return df
        
    else:
        recent_date = None
        df = None

    if recent_date is None:
        starting_date = start_date
    elif start_date < recent_date:
        starting_date = recent_date + timedelta(days=1)
    else:
        starting_date = start_date

    if get_function.__name__ == get_equity_data.__name__:
        data = get_function(symbol, start_date=starting_date, end_date=end_date, file_method=None, data_method='history')
    elif get_function.__name__ == get_bloomberg_data.__name__:
        data = get_function(symbol, start_date=start_date,end_date=end_date,file_method=None,max_page=int(sequence_length/1.5))
    else:
        data = get_function(symbol, start_date=starting_date, end_date=end_date, file_method=None)

    if not concat:
        df = data
    else:
        df = pd.concat([df, data], axis=0, ignore_index=True)

    j = df.to_json(orient='records', date_format='iso')
    if overwrite:
        with open(full_path, 'w') as f:
            f.write(j)

    return df






#--------------------------------------------------------------
#EQUITY DATA
#--------------------------------------------------------------
#gets closing prices from yfinance
def get_equity_data(symbol:str,start_date=datetime(2000,1,1),end_date=datetime(2023,6,1),api:str='yf',file_method='w',data_method='download'):
    start_date=datetime.strftime(start_date,'%Y-%m-%d')
    end_date=datetime.strftime(end_date,'%Y-%m-%d')

    if api == 'yf':
        if data_method=='download':
            data = yf.download(symbol,start=start_date,end=end_date)
            df = pd.DataFrame(data)
            df.reset_index(inplace=True)

        elif data_method=='history':
            ticker = yf.Ticker(symbol)
            try:
                data = ticker.history(period='1mo')
                df = pd.DataFrame(data)
                df.insert(loc=4,column='Adj Close',value=df['Close'])
                df.drop(columns=['Dividends', 'Stock Splits'],inplace=True)
                df.reset_index(inplace=True)
                df['Date'] = df['Date'].astype(str)
                df['Date'] = df['Date'].apply(lambda x: x[:10])
                df.loc[:,'Date'] = pd.to_datetime(df['Date'])
            except:
                data = yf.download(symbol,start=start_date,end=end_date)
                df = pd.DataFrame(data)
                df.reset_index(inplace=True)

        else:
            print('Invalid data method')
            return -1
        
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
def get_earnings(symbol:str,start_date=None,end_date=None,file_method='w',key='1C3O71BAB7HJXTWZ'):
    url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={key}'
    r = requests.get(url)
    j = r.json()

    try:
        dat = j['quarterlyEarnings']
    except:
        new_key = change_alphavantage_key(key)
        url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={new_key}'
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



def get_balance_sheet(symbol:str,start_date=None,end_date=None,file_method='w'):
    url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={symbol}&output_size=max&apikey={alphavantage_key}'
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data)

    if file_method is not None:
        with open(f'data_equity/{symbol}_balance_sheet.dat',file_method) as file:
            json.dump(data,file)
    return df


def get_cashflow(symbol:str,start_date=None,end_date=None,file_method='w'):
    url = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={symbol}&time_from=20050101T0000&apikey={alphavantage_key}'
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


def earnings_init(symbol:str='none',df=None,start_date=None,end_date=None):
    if symbol!='none':
        with open(f'data_equity/{symbol}_earnings.dat','r') as file:
            r = file.read()
        data = pd.read_json(r,convert_axes=True)
    else:
        data = df

    for index in data.index:
        if data.iloc[index,1] == 'None':
            data.drop(index=index,inplace=True)
        elif data.loc[index,'surprise']=='None':
            data.loc[index,'surprise'] = 0
    
    data.iloc[:, 1:] = data.iloc[:, 1:].astype(float)

    dframe = fn.data_fill(data,damping_columns=['surprise'],damping_constant=0.6,start_date=start_date,end_date=end_date)
    return dframe


def close_5d_init(symbol:str,start_date=None,end_date=None):
    with open(f'data_equity/{symbol}.dat','r') as f:
        r = f.read()
    close_dat = pd.read_json(r,convert_axes=True)
    #close_dat.loc[:,'Date'] = pd.to_datetime(close_dat['Date'])
    close_dat.sort_values(by='Date')

    close_dat['Close_5d'] = close_dat['Adj Close'].shift(-5)
    close_dat['Close_5d'] = (close_dat['Close_5d']-close_dat['Adj Close'])/close_dat['Adj Close']
    close_dat=close_dat[['Date','Close_5d']]
    close_dat=close_dat.iloc[:-5]
    df = fn.data_fill(close_dat,start_date=start_date,end_date=end_date)

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
def get_bloomberg_data(symbol,max_page:int=50,start_date=None,end_date=None,file_method='w'):
    df = {'Date':[],'Headline':[],'Summary':[]}
    for page in range(max_page):
        url = f"https://www.bloomberg.com/markets2/api/search?page={page}&query={symbol}&sort=time:desc"

        payload = {}
        headers = {
        'authority': 'www.bloomberg.com',
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        #'cookie': '_gcl_au=1.1.1258867486.1687890016; _pxvid=43f2ef4b-1517-11ee-9c98-4d76c2da6132; _rdt_uuid=1687890040318.60c499b1-45fd-4a4e-8f20-2f74cd36be33; professional-cookieConsent=new-relic|perimeterx-bot-detection|perimeterx-pixel|google-tag-manager|google-analytics|microsoft-advertising|eloqua|adwords|linkedin-insights; drift_aid=50f77712-c0e7-4dca-be00-1a3be2328a99; driftt_aid=50f77712-c0e7-4dca-be00-1a3be2328a99; _ga_NNP7N7T2TG=GS1.1.1690421511.1.1.1690421544.27.0.0; optimizelyEndUserId=oeu1691579311263r0.14078048370792384; _sp_v1_data=585912; _sp_su=false; _gid=GA1.2.899983733.1691579312; ccpaUUID=8ddf4e4d-c0f6-4360-a8b8-a8380fef7366; dnsDisplayed=true; ccpaApplies=true; signedLspa=false; bbgconsentstring=req1fun1pad1; _gcl_aw=GCL.1691579312.CjwKCAjw8symBhAqEiwAaTA__DO-uluShYNXs3DHq9qThKK2LjCpEYvmtIQC0s2nkstqwB9aO3_W5xoCYs8QAvD_BwE; _gcl_dc=GCL.1691579312.CjwKCAjw8symBhAqEiwAaTA__DO-uluShYNXs3DHq9qThKK2LjCpEYvmtIQC0s2nkstqwB9aO3_W5xoCYs8QAvD_BwE; bdfpc=004.7462060169.1691579312202; _reg-csrf=s%3AlyBG8tgXEI9gNgMwydkfiRDm.RlyuzWcn50CU7nh9OqEZ95DENiWW7I6ne4qIka7hhS8; pxcts=139f8f51-36a5-11ee-ab90-6964554c7146; _scid=150ea57a-2797-4865-a5c9-ae5f03bd0f1b; _fbp=fb.1.1691579312789.341111564; agent_id=5b10caf2-1327-404d-bdd9-a996d347e790; session_id=020044b7-d1ac-4592-878e-cec443ac6e02; session_key=79b794f0c70c696278e62a0cf85e93415508cbaa; gatehouse_id=9d7077e9-22af-491f-a236-042c63c2a8f0; geo_info=%7B%22countryCode%22%3A%22US%22%2C%22country%22%3A%22US%22%2C%22cityId%22%3A%225368361%22%2C%22provinceId%22%3A%225332921%22%2C%22field_p%22%3A%22E6A909%22%2C%22field_d%22%3A%22rr.com%22%2C%22field_mi%22%3A-1%2C%22field_n%22%3A%22hf%22%2C%22trackingRegion%22%3A%22US%22%2C%22cacheExpiredTime%22%3A1692184112798%2C%22region%22%3A%22US%22%2C%22fieldMI%22%3A-1%2C%22fieldN%22%3A%22hf%22%2C%22fieldD%22%3A%22rr.com%22%2C%22fieldP%22%3A%22E6A909%22%7D%7C1692184112798; _li_dcdm_c=.bloomberg.com; _lc2_fpi=b1166d620485--01h7czqv6gbrbhddx2qjmrg1mb; _gac_UA-11413116-1=1.1691579314.CjwKCAjw8symBhAqEiwAaTA__DO-uluShYNXs3DHq9qThKK2LjCpEYvmtIQC0s2nkstqwB9aO3_W5xoCYs8QAvD_BwE; _sctr=1%7C1691553600000; seen_uk=1; exp_pref=AMER; ln_or=eyI0MDM1OTMiOiJkIn0%3D; _cc_id=cb4452165a13807982d81de51e7402ec; panoramaId=7071f388f239d3ecb953ab1733d316d5393826fd6dfa307fa39b72341051eee0; panoramaIdType=panoIndiv; afUserId=044f5126-bd6b-48c9-b84d-1a6205bfd708-p; AF_SYNC=1691579323016; _sp_v1_p=192; _parsely_session={%22sid%22:4%2C%22surl%22:%22https://www.bloomberg.com/search?query=AAPL&page=3&sort=time:desc%22%2C%22sref%22:%22%22%2C%22sts%22:1691627981370%2C%22slts%22:1691625583270}; _parsely_visitor={%22id%22:%22pid=fb1a7e688db3d4e11dc092a353d07cce%22%2C%22session_count%22:4%2C%22last_session_ts%22:1691627981370}; _sp_v1_ss=1:H4sIAAAAAAAAAItWqo5RKimOUbLKK83J0YlRSkVil4AlqmtrlXQGVlk0kYw8EMOgNhaXkfSQGOiwGnzKYgF_3pWTZQIAAA%3D%3D; _sp_krux=true; euconsent-v2=CPwSZUAPwSZUAAGABCENDPCgAP_AAEPAABpYH9oB9CpGCTFDKGh4AKsAEAQXwBAEAOAAAAABAAAAABgQAIwCAEASAACAAAACGAAAIAIAAAAAEAAAAEAAQAAAAAFAAAAEAAAAIAAAAAAAAAAAAAAAAIEAAAAAAUAAEFAAgEAAABIAQAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgYtAOAAcAD8ARwA7gCBwEHAQgAiIBFgC6gGvAWUAvMBggDFgDwkBcABYAFQAMgAcABAADIAGgARAAjgBMACeAH4AQgAjgBSgDKgHcAd4A9gCTAEpAOIAuQBkgQAGARwAnYCmw0AEBcgYACApsRABAXIKgAgLkGQAQFyDoDQACwAKgAZAA4ACAAFwAMgAaABEACOAEwAJ4AXQAxAB-AFGAKUAZQA7wB7AEmAJSAcQA6gC5AGSDgAoAFwBHAEcAJ2ApshANAAWABkAFwATAAxACOAFKAMqAdwB3gEpAOoAsoBchAACARwlAMAAWABkADgARAAjgBMADEAI4AUYA7wDqAMkJAAgALgEcKQFAAFgAVAAyABwAEAANAAiABHACYAE8AMQAfgBRgClAGUAO8AlIB1AFyAMkKABAALgAyAIOAps.YAAAAAAAAAAA; consentUUID=87aa1011-a199-4670-9bfa-e8a090dc0452_22; country_code=US; _reg-csrf-token=qtPaStnF-joAKNZrqCPnkMRvrKeSRd3V9FQQ; _user-data=%7B%22status%22%3A%22anonymous%22%7D; _last-refresh=2023-8-10%201%3A2; _scid_r=150ea57a-2797-4865-a5c9-ae5f03bd0f1b; __sppvid=43fc3cd7-9fee-4242-ad89-772dd26f0988; _uetsid=13d2691036a511ee9072e745523f4a97; _uetvid=524378e0151711eeb02ba54b6db7a142; _px3=006a2178000e88b5a8383e394d20abe4dec20fe6b83fec951bb5462fd849eb60:FwQbebtd0GCPQQcjQh6Az/v7P3oymeKuiyEhxRdWi+jkaqslxMxMp6XHIFTbdnbVR18c+tpQWFgGzXZeME4ARw==:1000:6mHvOZ7bghHs9Vwo3KeoDBalw4QnYCNkRQ86Wex8kkb/kSAhQS7Ez3WTj+2sp8oIfTypK/JaqIz6uRQ0XEV+riKpp5m6hFvxfiXPlKY2/RUzrecx5lF5mVDetdpb2t+MFvJp7X7XZTaxejFyjHlzxShVPX0Rec9UipbH7A9OdQVnff7EbL7t+6oJc2BXDTPSHOLsISghRszRcWl9R/9eNQ==; _px2=eyJ1IjoiOTY0ZTJkODAtMzcxOS0xMWVlLWJlYTUtYmZhODc3YWIwYmY4IiwidiI6IjQzZjJlZjRiLTE1MTctMTFlZS05Yzk4LTRkNzZjMmRhNjEzMiIsInQiOjE2OTE2Mjk2NTM4NTQsImgiOiIxYWU0NTYwNWNhOWU0OWRlMGM3NmQyZWYyN2Y2ZTlmNjA0NzE3ZWQ2NDdmMGRiMTIwMGY3ZDAyYjFkOWQ2Y2IyIn0=; panoramaId_expiry=1692234153858; __gads=ID=76572d94ef31506c:T=1691579321:RT=1691629354:S=ALNI_MYlt1TkN2cOqMPx-hpXB8MARppYBQ; __gpi=UID=000009b26c3c0a8a:T=1691579321:RT=1691629354:S=ALNI_MbuI-GU0H0AoY52S8l18G7ro2rJQA; _ga=GA1.2.1454142836.1687890016; _pxde=fc60b9dadff19404c07f24b80fdb9b6d6a574085838e971e8f4b42cf3d6b5154:eyJ0aW1lc3RhbXAiOjE2OTE2Mjk0ODc5NjgsImZfa2IiOjAsImlwY19pZCI6W119; _gat_UA-11413116-1=1; _ga_GQ1PBLXZCT=GS1.1.1691627960.4.1.1691629514.56.0.0; exp_pref=AMER; country_code=US',
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
            #time.sleep(5)
            print('bloomberg except block ran')
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
            terminate_and_run_proton(proton_path)
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
def get_barrons_data(company,max_page:int=60,start_date=datetime(2016,1,1),end_date=datetime(2023,6,1),time_interval:int=4,file_method='w'):
    
    df = {'Date':[],'Headline':[],'Summary':[]}
    current_date = start_date
    counter = 0
    page = 0
    #terminate_and_run_proton(r"C:\Program Files\Proton\VPN\v3.1.0\ProtonVPN.exe",terminate=False)
    #terminate_and_run_proton(r"D:\Program Files (x86)\Proton Technologies\ProtonVPN\ProtonVPN.exe",terminate=False)

    while current_date<=end_date:
        current_year,current_month,current_day = IntervalDate(current_date,time_interval).get_current_date()
        next_year,next_month,next_day = IntervalDate(current_date,time_interval).get_next_date()
        current_date_str = datetime.strftime(current_date,'%Y-%m-%d')
        next_date_str = datetime.strftime(current_date+timedelta(days=time_interval),'%Y-%m-%d')

        url = f"https://www.barrons.com/search?id=%7B%22query%22%3A%7B%22not%22%3A%5B%7B%22terms%22%3A%7B%22key%22%3A%22SectionType%22%2C%22value%22%3A%5B%22NewsPlus%22%5D%7D%7D%5D%2C%22and%22%3A%5B%7B%22terms%22%3A%7B%22key%22%3A%22languageCode%22%2C%22value%22%3A%5B%22en%22%2C%22en-us%22%5D%7D%7D%2C%7B%22date%22%3A%7B%22key%22%3A%22liveDate%22%2C%22value%22%3A%22{next_date_str}T18%3A59%3A59-05%3A00%22%2C%22operand%22%3A%22LessEquals%22%7D%7D%2C%7B%22date%22%3A%7B%22key%22%3A%22liveDate%22%2C%22value%22%3A%22{current_date_str}T00%3A00%3A00%2B00%3A00%22%2C%22operand%22%3A%22GreaterEquals%22%7D%7D%2C%7B%22terms%22%3A%7B%22key%22%3A%22Product%22%2C%22value%22%3A%5B%22Barrons.com%22%2C%22Barrons.com%20Automated%20Market%20Wraps%22%2C%22Barrons%20Blogs%22%2C%22Barrons%20Advisor%20Credits%20Video%22%2C%22Barrons%20Broadband%20Video%22%2C%22Barrons%20Summit%20Video%22%2C%22Barrons%20Video%20Live%20Q%26A%22%2C%22Barrons.com%20Webstory%22%2C%22Barrons%20Live%20Coverage%22%5D%7D%7D%5D%2C%22or%22%3A%5B%7B%22query_string%22%3A%7B%22value%22%3A%22{company}%22%2C%22default_or_operator%22%3Atrue%2C%22parameters%22%3A%5B%7B%22property%22%3A%22headline%22%2C%22boost%22%3A3%7D%2C%7B%22property%22%3A%22keywords%22%2C%22boost%22%3A4%7D%2C%7B%22property%22%3A%22byline%22%2C%22boost%22%3A3%7D%2C%7B%22property%22%3A%22body%22%2C%22boost%22%3A4%7D%2C%7B%22property%22%3A%22section_name%22%2C%22boost%22%3A5%7D%5D%7D%7D%2C%7B%22full_text%22%3A%7B%22value%22%3A%22{company}%22%2C%22match_phrase%22%3Atrue%2C%22parameters%22%3A%5B%7B%22property%22%3A%22headline%22%2C%22boost%22%3A3%7D%2C%7B%22property%22%3A%22body%22%2C%22boost%22%3A4%7D%5D%7D%7D%5D%7D%2C%22sort%22%3A%5B%7B%22key%22%3A%22relevance%22%2C%22order%22%3A%22desc%22%7D%5D%2C%22count%22%3A20%7D%2Fpage%3D{page}&type=allesseh_search_full_v2"

        payload = {}
        headers = {
        'authority': 'www.barrons.com',
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        #'cookie': 'wsjregion=na%2Cus; ab_uuid=57746ccb-f0e2-4456-acf3-5bf962e1ecd9; usr_bkt=1p6sdUNJWD; _pcid=%7B%22browserId%22%3A%22ll52818318ms5m7h%22%7D; cX_P=ll52818318ms5m7h; _pctx=%7Bu%7DN4IgrgzgpgThIC4B2YA2qA05owMoBcBDfSREQpAeyRCwgEt8oBJAEzIE4AmHgZi4CsvAIwB2DqIAMADkHTRAkAF8gA; _pubcid=a9c66174-d226-4d66-96b3-b1de5cdb1c63; cX_G=cx%3Aaay94xhyxbv11gicm2zvcbs06%3A118c0xph4rjn9; ccpaUUID=c2d2144e-67e6-4ffb-a000-5fce7bc58548; _rdt_uuid=1691665942919.a6e1c4be-6736-4b59-bb15-f8bde9c1518d; _fbp=fb.1.1691665943068.1240242682; _cls_v=451db5d3-3a0a-4094-b5a0-40ddf833857a; _ncg_domain_id_=0ec0072f-4db3-49d7-ba37-b2560556712d.1.1691665942904.1754737942904; _ncg_g_id_=4d40a8c1-d098-4817-9b73-adeb1ac41c21.3.1691665459.1754737942904; _lr_env_src_ats=false; _dj_sp_id=efedd005-ecfa-4efd-b74f-0f17486efade; permutive-id=6c6c2bca-c05c-4138-88d6-6e1f360c4fc5; optimizelyEndUserId=oeu1691666193209r0.5358536663461277; dnsDisplayed=false; signedLspa=false; _ncg_id_=0ec0072f-4db3-49d7-ba37-b2560556712d; _parsely_visitor={%22id%22:%22pid=5e443952-8844-4822-9928-a578fd3058a9%22%2C%22session_count%22:2%2C%22last_session_ts%22:1691670954771}; consentUUID=d76965c7-8539-4c00-acd4-269856889512_22; AMCV_CB68E4BA55144CAA0A4C98A5%40AdobeOrg=1585540135%7CMCIDTS%7C19580%7CMCMID%7C07811295556865454993086478630926961519%7CMCAAMLH-1692279689%7C9%7CMCAAMB-1692279689%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1691682089s%7CNONE%7CMCAID%7CNONE%7CvVersion%7C4.4.0; _gcl_au=1.1.499596267.1691674896; utag_main=v_id:0189df25c56a0089c6625ae32bf80506f001806700aee^$_n:3^$_s:0^$_t:1691677961776^$vapi_domain:barrons.com^$_revpage:BOL_ResearchTools_Barrons%20Search%3Bexp-1691679761781^$ses_id:1691674889255%3Bexp-session^$_n:7%3Bexp-session; _ncg_sp_id.e48a=0ec0072f-4db3-49d7-ba37-b2560556712d.1691665943.13.1691676162.1691676157.b8146b02-6caf-4b7c-92a7-83893fcf9f42; cto_bundle=pCG-n19EQURaY0RhRXRScURXR3N2QmQ3V1lvNHNEcmU2RlNtR3c1WDVmQjBsVWJHWm5ENnhvY0g2dU9GNlUxeHd0b2QwWWFlR2t1RzBxbiUyQmllaGFndDJ1RVJRcnljZDNvMXJ4NzgzbGV5amsyVXdWVkFkcUhsTnl2OEFuRGRJOHp5NEJXODNpWiUyQkFXUUNGcFFWZ0tCMWJhWnh3JTNEJTNE; __gads=ID=c0815736f6fedddf:T=1691665942:RT=1691676182:S=ALNI_MboodYppO-KeovmgXjnvWNX5fiBHw; __gpi=UID=000009b291b15d47:T=1691665942:RT=1691676182:S=ALNI_MaJtq0rnOlFeh7f7piV5qu5GnNPAA; DJSESSION=country%3Dus%7C%7Ccontinent%3Dna%7C%7Cregion%3Dca%7C%7Ccity%3Dlosangeles%7C%7Clatitude%3D33.9733%7C%7Clongitude%3D-118.2487%7C%7Ctimezone%3Dpst%7C%7Czip%3D90001-90068%2B90070-90084%2B90086-90089%2B90091%2B90093-90096%2B90099%2B90189; gdprApplies=false; ccpaApplies=true; vcdpaApplies=false; regulationApplies=gdpr%3Afalse%2Ccpra%3Atrue%2Cvcdpa%3Afalse; usr_prof_v2=eyJwIjp7InBzIjowLjA1NzUxLCJxIjowLjU2fSwiaWMiOjR9; _lr_geo_location_state=CA; _lr_geo_location=US; _pbjs_userid_consent_data=6683316680106290; _sp_su=false',
        'newrelic': 'eyJ2IjpbMCwxXSwiZCI6eyJ0eSI6IkJyb3dzZXIiLCJhYyI6IjE2ODQyNzMiLCJhcCI6IjEzODU5MTI0NzkiLCJpZCI6IjkxMWIwZmMzMzkyZjc1MmEiLCJ0ciI6IjI5N2U4ZTcwOTg4YzE4ZDgxNmRiMTkyYTk4ZDI1MDAwIiwidGkiOjE2OTM5NTkyNDQ1NjcsInRrIjoiMTAyMjY4MSJ9fQ==',
        'referer': f'https://www.barrons.com/search?query={company}&quotequery={company}&isToggleOn=true&operator=OR&sort=relevance&duration=2d&startDate={current_year}%2F{current_month}%2F{current_day}&endDate={next_year}%2F{next_month}%2F{next_day}&source=barrons%2Cbarronsblog%2Cbarronsvideos%2Cbarronswebstory%2Cbarronslivecoverage',
        'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        'traceparent': '00-297e8e70988c18d816db192a98d25000-911b0fc3392f752a-01',
        'tracestate': '1022681^@nr=0-1-1684273-1385912479-911b0fc3392f752a----1693959244567',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
        }
        try:
            response = requests.request("GET", url, headers=headers, data=payload)
            data = response.json()
        except:
            terminate_and_run_proton(proton_path)
            continue        
        
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
            #'cookie': 'wsjregion=na%2Cus; ab_uuid=57746ccb-f0e2-4456-acf3-5bf962e1ecd9; usr_bkt=1p6sdUNJWD; _pcid=%7B%22browserId%22%3A%22ll52818318ms5m7h%22%7D; cX_P=ll52818318ms5m7h; _pctx=%7Bu%7DN4IgrgzgpgThIC4B2YA2qA05owMoBcBDfSREQpAeyRCwgEt8oBJAEzIE4AmHgZi4CsvAIwB2DqIAMADkHTRAkAF8gA; _pubcid=a9c66174-d226-4d66-96b3-b1de5cdb1c63; cX_G=cx%3Aaay94xhyxbv11gicm2zvcbs06%3A118c0xph4rjn9; ccpaUUID=c2d2144e-67e6-4ffb-a000-5fce7bc58548; _rdt_uuid=1691665942919.a6e1c4be-6736-4b59-bb15-f8bde9c1518d; _fbp=fb.1.1691665943068.1240242682; _cls_v=451db5d3-3a0a-4094-b5a0-40ddf833857a; _ncg_domain_id_=0ec0072f-4db3-49d7-ba37-b2560556712d.1.1691665942904.1754737942904; _ncg_g_id_=4d40a8c1-d098-4817-9b73-adeb1ac41c21.3.1691665459.1754737942904; _lr_env_src_ats=false; _dj_sp_id=efedd005-ecfa-4efd-b74f-0f17486efade; permutive-id=6c6c2bca-c05c-4138-88d6-6e1f360c4fc5; optimizelyEndUserId=oeu1691666193209r0.5358536663461277; dnsDisplayed=false; signedLspa=false; _ncg_id_=0ec0072f-4db3-49d7-ba37-b2560556712d; _parsely_visitor={%22id%22:%22pid=5e443952-8844-4822-9928-a578fd3058a9%22%2C%22session_count%22:2%2C%22last_session_ts%22:1691670954771}; consentUUID=d76965c7-8539-4c00-acd4-269856889512_22; AMCV_CB68E4BA55144CAA0A4C98A5%40AdobeOrg=1585540135%7CMCIDTS%7C19580%7CMCMID%7C07811295556865454993086478630926961519%7CMCAAMLH-1692279689%7C9%7CMCAAMB-1692279689%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1691682089s%7CNONE%7CMCAID%7CNONE%7CvVersion%7C4.4.0; _gcl_au=1.1.499596267.1691674896; utag_main=v_id:0189df25c56a0089c6625ae32bf80506f001806700aee$_sn:3$_ss:0$_st:1691677961776$vapi_domain:barrons.com$_prevpage:BOL_ResearchTools_Barrons%20Search%3Bexp-1691679761781$ses_id:1691674889255%3Bexp-session$_pn:7%3Bexp-session; _ncg_sp_id.e48a=0ec0072f-4db3-49d7-ba37-b2560556712d.1691665943.13.1691676162.1691676157.b8146b02-6caf-4b7c-92a7-83893fcf9f42; cto_bundle=pCG-n19EQURaY0RhRXRScURXR3N2QmQ3V1lvNHNEcmU2RlNtR3c1WDVmQjBsVWJHWm5ENnhvY0g2dU9GNlUxeHd0b2QwWWFlR2t1RzBxbiUyQmllaGFndDJ1RVJRcnljZDNvMXJ4NzgzbGV5amsyVXdWVkFkcUhsTnl2OEFuRGRJOHp5NEJXODNpWiUyQkFXUUNGcFFWZ0tCMWJhWnh3JTNEJTNE; __gads=ID=c0815736f6fedddf:T=1691665942:RT=1691676182:S=ALNI_MboodYppO-KeovmgXjnvWNX5fiBHw; __gpi=UID=000009b291b15d47:T=1691665942:RT=1691676182:S=ALNI_MaJtq0rnOlFeh7f7piV5qu5GnNPAA; DJSESSION=country%3Dus%7C%7Ccontinent%3Dna%7C%7Cregion%3Dca%7C%7Ccity%3Dlosangeles%7C%7Clatitude%3D33.9733%7C%7Clongitude%3D-118.2487%7C%7Ctimezone%3Dpst%7C%7Czip%3D90001-90068%2B90070-90084%2B90086-90089%2B90091%2B90093-90096%2B90099%2B90189; gdprApplies=false; ccpaApplies=true; vcdpaApplies=false; regulationApplies=gdpr%3Afalse%2Ccpra%3Atrue%2Cvcdpa%3Afalse; usr_prof_v2=eyJwIjp7InBzIjowLjA1NzUxLCJxIjowLjU2fSwiaWMiOjR9; _lr_geo_location_state=CA; _lr_geo_location=US; _pbjs_userid_consent_data=6683316680106290; _sp_su=false',
            'newrelic': 'eyJ2IjpbMCwxXSwiZCI6eyJ0eSI6IkJyb3dzZXIiLCJhYyI6IjE2ODQyNzMiLCJhcCI6IjEzODU5MTI0NzkiLCJpZCI6IjFiYTgxODViOTY2MzI2MjIiLCJ0ciI6ImM3M2FlZjI1NjY1MTkwZGE5MGQ3OGM1ZmEzNDE3OTAwIiwidGkiOjE2OTM5NTkyNDUxMjUsInRrIjoiMTAyMjY4MSJ9fQ==',
            'referer': f'https://www.barrons.com/search?query={company}&quotequery={company}&isToggleOn=true&operator=OR&sort=relevance&duration=2d&startDate={current_year}%2F{current_month}%2F{current_day}&endDate={next_year}%2F{next_month}%2F{next_day}&source=barrons%2Cbarronsblog%2Cbarronsvideos%2Cbarronswebstory%2Cbarronslivecoverage',
            'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'traceparent': '00-c73aef25665190da90d78c5fa3417900-1ba8185b96632622-01',
            'tracestate': '1022681@nr=0-1-1684273-1385912479-1ba8185b96632622----1693959245125',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
            }

            try:
                id_response = requests.request("GET", url, headers=id_headers, data=id_payload)
                id_data = id_response.json()
            except:
                print('An id_response error occurred')
                continue

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
            timestamp_str = datetime.strftime(timestamp_date,'%Y-%m-%d')
            print(timestamp_str)
            df['Date'].append(timestamp_str)
            df['Headline'].append(headline)
            df['Summary'].append(summary)
        
        current_date = current_date + timedelta(days=time_interval+1)

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

    terminate_and_run_proton(proton_path,terminate=False)
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
            terminate_and_run_proton(proton_path)
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
        #print(f'closest date: {date}')
        data = data.loc[data['Date']>=date]
    if end_date is not None:
        date = fn.find_closest_date(data,end_date,direction='right')
        data = data.loc[data['Date']<=date]
    data.reset_index(inplace=True,drop=True)
    sentiment_df = {'Headline_score':[]}

    for index in data.index:
        headline_score = roberta_sentiment(data['Headline'].iloc[index])
        #summary_score = roberta_sentiment(data['Summary'].iloc[index])
        sentiment_df['Headline_score'].append(headline_score)
        #sentiment_df['Summary_score'].append(summary_score)
        if index%1000==0 and index>0:
            print(index)
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
        try:
            df_list.append(news_roberta(symbol,outlet,start_date=start_date,end_date=end_date))
        except:
            print(f'could not format {symbol} {outlet} data')

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
    df['Date'] = pd.to_datetime(df['Date'])
    # Format "Date" column as 'yyyy-mm-dd' (year-month-day)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df = fn.data_fill(df,interpolate_columns='all',start_date=start_date,end_date=end_date)
    j = df.to_json(orient='records',date_format='iso')
    if file_method is not None:
        with open(f'data_misc/{file_prefix}_{symbol}.dat',file_method) as file:
            file.write(j)
    print(f'Successfully formatted {symbol} news data')
    return df


#going to keep news_init normal (meaning passing the file path instead of the data)
def news_init(symbol:str,df=None,prefix='news_sentiment',start_date=None,end_date=None):
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



def trend_init(symbol:str='none',df=None,start_date=None,end_date=None):
    if symbol == 'none' and df is None:
        print('trend_init Error: symbol and df cannot both be empty')
    
    if symbol!='none':
        with open(f'data_misc/trend_{symbol}.dat', 'r') as file:
            data = file.read()
        trend_df = pd.read_json(data,convert_dates=True)
    else:
        trend_df = df
    
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
'''
cik = {'AAL':6201,'AAPL':320193,'AMD':2488,
       'AMZN':1018724,'BAC':70858,'BANC':1169770,'BRK-B':1067983,'CGNX':851205,
       'CSCO':858877,'DELL':1571996,'DIS':1744489,'F':37996,'GE':40545,
       'GOOG':1652044,'INTC':50863,'MCD':63908,'META':1326801,'MLM':916076,'MSFT':789019,'NFLX':1065280,
       'NVDA':1045810,'QCOM':804328,'ROKU':1428439,'RUN':1469367,'SBUX':829224,'SHOP':1594805,
       'T':732717,'TGT':27419,'TSLA':1318605,'UPS':1090727,'WMT':104169}
cik_df = pd.DataFrame(cik)
'''
sec_data = pd.read_csv('csv_data/SEC_CODES.csv')

def get_insider_trading_data(symbol:str,max_page:int=6,start_date=datetime(2005,1,1),end_date=datetime(2023,6,1),file_method='w'):
    #I'm thinking about doing percentage of securities owned vs acquired or sold
    pages = []
    for n in range(max_page):
        pages.append(80*n) #the sec includes 80 rows of data per page

    df = {'Date':[],'InsiderFlow':[]} #change will be negative for a sell, positive for a purchase
    cik_code = sec_data.loc[sec_data['ticker']==symbol]
    cik_code = cik_code['cik'].iloc[-1]
    locator = str(cik_code).zfill(10)
    #print(locator)

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
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
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


def insider_init(symbol:str='none',df=None,start_date=None,end_date=None):
    if symbol=='none' and df is None:
        print('insider_init Error, can\'t have empty symbol and df data')
    
    if symbol!='none':
        with open(f'data_misc/insider_{symbol}.dat','r') as file:
            r = file.read()
        data = pd.read_json(r,convert_axes=True)
    else:
        data = df
    #replace all rows where insider_flow==1 to =0
    data['InsiderFlow'] = data['InsiderFlow'].replace(1,0)
    df = fn.data_fill(data,damping_columns=['InsiderFlow'],damping_constant=5.0,start_date=start_date,end_date=end_date)
    return df








#--------------------------------------------------------------------------------------
#UNUSUAL OPTIONS FLOW AND PRICE TARGETS
#--------------------------------------------------------------------------------------

def get_options_flow(symbol,start_date=datetime(2016,1,1),end_date=datetime(2023,6,1),file_method='w'):
    current_date = start_date
    current_str = datetime.strftime(current_date,'%Y-%m-%d')

    df = {'Date':[],'vOIR':[],'volatility':[],'delta':[]} #vOIR stands for volume open interest ratio. The higher the abs(vOIR), the more unusual the flow is
    prev_volatility = 0

    while current_date<=end_date:
        #there won't be any options flow data on weekends
        if current_date.weekday()>4:
            current_date = current_date + timedelta(days=1)
            current_str = datetime.strftime(current_date,'%Y-%m-%d')
            continue
            
        url = f"https://www.barchart.com/proxies/core-api/v1/lists-snapshot/get?list=stocks.us.unusual_options&fields=symbol%2CbaseSymbol%2CbaseLastPrice%2CbaseSymbolType%2CsymbolType%2CstrikePrice%2CexpirationDate%2CdaysToExpiration%2CbidPrice%2Cmidpoint%2CaskPrice%2ClastPrice%2Cvolume%2CopenInterest%2CvolumeOpenInterestRatio%2Cvolatility%2Cdelta%2CtradeTime%2CsymbolCode&orderBy=volumeOpenInterestRatio&orderDir=desc&in(dateOnList%2C({current_str}))=&in(symbolType%2C(Call%2CPut))=&in(expirationType%2C(monthly%2Cweekly))=&in(baseSymbol%2C({symbol}))=&limit=100&page=1&raw=1&meta=field.shortName%2Cfield.type%2Cfield.description%2Clists.lastUpdate"

        payload = {}
        headers = {
        'authority': 'www.barchart.com',
        'accept': 'application/json',
        'accept-language': 'en-US,en;q=0.9',
        'cookie': '_gcl_au=1.1.239753414.1694226957; usprivacy=1YNY; _admrla=2.2-3bd8bb55f72c3979-e4e3ce55-4eb6-11ee-96e3-65bbce279b88; alo_uid=88940fbf-d7fb-4297-9569-d05811571d1b; _gd_visitor=5aeb8186-1c10-4c03-8901-08e7db64d0bb; _gd_svisitor=4d4036175f190000f4cb6663cf01000093725002; __browsiUID=e5c0743e-a1cc-4dc4-83eb-3c2999ec742c; _ga_4HQ9CY2XKK=GS1.1.1694235995.1.1.1694236036.19.0.0; bcFreeUserPageView=0; _gid=GA1.2.929628719.1695006544; _gd_session=59d087fc-3a08-494f-823c-0ce0d027dd43; __browsiSessionID=54354737-d3d3-45dd-99c5-f286bbeb5172&true&false&SEARCH&us&desktop-4.18.5&false; _ga_W0HSBQPE0S=GS1.1.1695006545.1.0.1695006546.0.0.0; cto_bundle=adCVpl8zSkdjYmdweGV1UGtWQ0FyZWFlb3VFWGk0MXA0QjZ0ODBQYW9YJTJCZ3UyQUQlMkIlMkJyUDkzbmw4c1Y1T0lMMmt6UTljaTVUWnZYak8wWUElMkJrWWFJdnpsdzQxJTJGMktGYXhualJTQURyN0d4NWUlMkJIUllRODF4NHA4MnM2YUE0ZzQlMkZER1B6ZGNPejJBM1d1V0t3RndQMG1VNmcxYXlFWTNkeE9mamFXNVRKa0UyRjc1d1AlMkJrTWF1NlJSY1RyRDZvSDdjMmgyRVlPcUNtcEhHNUZWNzdxSWwlMkZseTd3JTNEJTNE; webinar158WebinarClosed=true; IC_ViewCounter_www.barchart.com=3; _awl=2.1695006761.5-51e83fc2fc729d987337916c0e78ce70-6763652d75732d7765737431-0; _ga_PXEQ7G89EH=GS1.2.1695006548.2.0.1695006765.0.0.0; _hjSessionUser_2563157=eyJpZCI6ImJlZGE5NDc5LWU1MGEtNWRmYi05MjU0LWRjOTBlZDYwMTQ1NyIsImNyZWF0ZWQiOjE2OTQyMzQzNjkyMjksImV4aXN0aW5nIjp0cnVlfQ==; _hjSession_2563157=eyJpZCI6ImQ5M2I2N2U4LTkzODItNGRmYi1hNjA1LTVmMTZmZWM3OGVmZSIsImNyZWF0ZWQiOjE2OTUwMDY3ODg5ODYsImluU2FtcGxlIjpmYWxzZX0=; _hjAbsoluteSessionInProgress=0; remember_web_59ba36addc2b2f9401580f014c7f58ea4e30989d=eyJpdiI6IllVeVp0NVMraG9uQVhLTS9sMEJXM1E9PSIsInZhbHVlIjoiZmZiTDhuT1dGb1ZZS2s3NVpJM1c2WU85dFdZd3dITU5XQ3NEbG1YQlcwOWR0MXdVNWx1SE9FRENWWldxbG1wUnpGNkc3UlJzMlUwcXdjem1QaU5kZS9oMmc1eEcxT1lPS2djZWw2dGl6Q3ZraVlGUFI1d3FuN3p5Z1VQNWlSMTI1Q1VqQm5pN1ZKWWpYcTBoNjNSTEZ1RmdLVEJxOGV3Sk5keUhkS2tOS3R6M0gwTE9WNkE4N3htazlJeWdTYkdvODMzSUthWk50TlREWW1OeEdOd1ZGVjU3QnprbjBqZ3lJZ2h3bEdIMjIvND0iLCJtYWMiOiI2ZWQ5ZWNjNGIxNTI4ODJlNjQ4NzQyNzcxZThmMDExMDhmYzY4MDhiNmExNzcyZDE4MzI4N2M2ZjI4NTJlYjJiIn0%3D; market=eyJpdiI6IjlGSzNHN0xBSmR4SFJKQmtaT2J0d2c9PSIsInZhbHVlIjoiMUJxRjFYQVYreGhYeGRJMVkrMkFRbzl6M0IvSkE5U3NHRHFJV25Ddy9oSDRxSDV6YnJHdStVYUZtYnIvRlVqSyIsIm1hYyI6ImI1MjRiN2RhYTIwOWI5Yjc3OTQ2M2I2ZDljYzk3NTU5ZTM2YmIwOTM2N2UzYjdlZGNjNmMwMjI1MDkyYzFlOTcifQ%3D%3D; laravel_token=eyJpdiI6IkI3RlExZkMreERJYnI1RDY2SExSb1E9PSIsInZhbHVlIjoiT1JHRnF1SjlwV0NrMUdJSHFKYnVZL282RDBDN1FpVmlKdXFLbzQ0MFFOaG5xcUR0enNzWkN6WTgvdkRtUXhpeXUvcFJCTktQSmFrYWI3bytLeGlDNXBqeWk3a25UWk0va2FuV3dtTEtCa1R3Q2xMdXRja2R6clNnWkVaZUdmM0tUZnQ5ZHJxQ1JDNXhma0tQUk5pSENwMThpOUFnVWN3UUFmUDYvdjRuWEpPdkJhd2pObTl5cFI0bWxiMWRBeUhOZmZjM2NPTkRnVEUvcHFzMHRxWmZLQUU2MWcxUUhGcEQ5V3dwNGRBclFzeERuN2lpL3pVM2o1czl3YWRUY0FIMHVuS2w1NHpZTVhvQUV5b210MVFib2J3MEwwcjJkRUVKSHJLdU9FY2w3N0VaeklKeUtLQkdkTEoxRE5XV3NVaHUiLCJtYWMiOiJmYmUyMDA0NmNhNjZjMjQwMGMxMjlmZjU5NzljNWFhODI2ZmRjNzU4MjBiZDRjMzRjOTIxMTM5YTk4YzhhNDllIn0%3D; XSRF-TOKEN=eyJpdiI6IjNuSDZwbGNyOUg2L1dTZFFtTmF0cXc9PSIsInZhbHVlIjoiSXVMalNWM1Z6eS90QVBtTWplQzJqT0xVeEF6Z2hRM2I5NFNQYlZQOTJidFp5emZIbU9qelhIVFZJMUQ3clpRTXNSWlluenlMRVpwK0ZLN1lFdW5lYngzY2tqNlF3c2Rtb1hESTR1TWxadjR6OXJhdEhLNWh2UHJSWDYyMXpUSWUiLCJtYWMiOiIwMWFkOTFiMjYyMTkwMWI5ZjU3MWQxNWEyZTgwYzcyNWM3NDBlMTI5YWVlMjA3OTVmYWQ4ZWNjNGMzYjhlZDdhIn0%3D; laravel_session=eyJpdiI6IjI5bENqK2RUU2lKb3B6L29Id2IzUlE9PSIsInZhbHVlIjoiVENzMUF1R1IzL1ZHMzFIUFJlSWRrcy9zQTNhSkhzOHowSkxEUFFvdVJXLy9zakQrUmNkYkh2VnhVaytvOVVQYUR6VTByaDNVZGI0S2tveXlMNXhXK3NkdFJxdUgycVJiOUlGUlZpbG1rbk1aNTJReTJ4VXZzWFdIZzN5YkI4b3UiLCJtYWMiOiIzNzM2N2MwOTIxZjk0Yzk5NGYyZTZjMTJiYTFjZTgxN2JiZjk0N2ZkOGNmMzU3MzQ5YWEyNzY2ODFkODRkOTAyIn0%3D; _ga=GA1.2.1577042263.1694226957; unusualFilter=%7B%7D; __gads=ID=143414b2c8014ab6-221365c9c6e3003e:T=1694226955:RT=1695007224:S=ALNI_MZZ8agNptiDLliyUrTUDfzcJr4HSA; __gpi=UID=00000d9129b77101:T=1694226955:RT=1695007224:S=ALNI_Mbae-5YnVwTICOVbgcixHPkk_3hFg; _ga_PE0FK9V6VN=GS1.1.1695006544.4.1.1695007294.50.0.0; unusualFilterParams=%7B%22expirationType%22%3A%5B%22monthly%22%2C%22weekly%22%5D%2C%22symbolType%22%3A%5B%22Call%22%2C%22Put%22%5D%2C%22symbols%22%3A%22A%22%7D',
        'referer': f'https://www.barchart.com/options/unusual-activity/stocks?type=historical&historicalDate={current_str}&useFilter=1',
        'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-origin',
        #'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36',
        'user-agent':'Mozilla/5.0 (X11; U; Linux 2.4.2-2 i586; en-US; m18) Gecko/20010131 Netscape6/6.01',
        'x-xsrf-token': 'eyJpdiI6IjNuSDZwbGNyOUg2L1dTZFFtTmF0cXc9PSIsInZhbHVlIjoiSXVMalNWM1Z6eS90QVBtTWplQzJqT0xVeEF6Z2hRM2I5NFNQYlZQOTJidFp5emZIbU9qelhIVFZJMUQ3clpRTXNSWlluenlMRVpwK0ZLN1lFdW5lYngzY2tqNlF3c2Rtb1hESTR1TWxadjR6OXJhdEhLNWh2UHJSWDYyMXpUSWUiLCJtYWMiOiIwMWFkOTFiMjYyMTkwMWI5ZjU3MWQxNWEyZTgwYzcyNWM3NDBlMTI5YWVlMjA3OTVmYWQ4ZWNjNGMzYjhlZDdhIn0='
        }

        response = requests.request("GET", url, headers=headers, data=payload)
        j = response.json()

        vOIR = 0
        delta = 0
        volatility = prev_volatility
        date = current_str
        n = 0 #counter to perform averages

        for data in j['data']:
            try:
                voir_temp = float(data["volumeOpenInterestRatio"])
                if data["symbolType"] == 'Put':
                    voir_temp*=-1
                vOIR += voir_temp
                #print(data["delta"])
                #print(data["volatility"])
                delta += float(data["delta"])
                volatility += float(data["volatility"].strip("%"))
                n+=1
            except:
                print('except block ran')
                continue

        if n>0:
            delta/=n
            volatility/=n
            prev_volatility = volatility
            vOIR/=n

        df['Date'].append(date)
        df['delta'].append(delta)
        df['vOIR'].append(vOIR)
        df['volatility'].append(volatility)                

        current_date = current_date + timedelta(days=1)
        current_str = datetime.strftime(current_date,'%Y-%m-%d')
        if current_date.day == 1:
            print(current_str)

    df = pd.DataFrame(df)
    json_data = df.to_json(orient='records',date_format='iso')

    if file_method is not None:
        with open(f'data_misc/optionsflow/{symbol}.dat',file_method) as f:
            f.write(json_data)

    return df


#appends a single date to all of the options flow data. Make sure to run this daily. If behind, run functions in unusualoptions.py
def optionsflow_appender(date):
    current_date = date    
    current_str = datetime.strftime(current_date,'%Y-%m-%d')

    df_dict = {}

    try:
        options_df = pd.read_csv(f'data_misc/optionsflow/Unusual-Stock-Options-Activity-{current_str}.csv')
    except Exception as e:
        current_date = current_date + timedelta(days=1)
        current_str = datetime.strftime(current_date,'%Y-%m-%d')
        print('options appender error: must download recent file')

    for index,value in options_df.iterrows():
        try:
            volOI = float(value['Vol/OI'])
        except:
            volOI = float(value['Vol/OI'].replace(',',''))
        try:
            iv = float(value['IV'].strip('%'))/100
        except:
            iv = str(value['IV']).strip('%')
            iv = iv.replace(',','')
            iv = float(iv)

        delta = float(value['Delta'])
        symbol = value['Symbol']

        if value['Type'] == 'Put':
            volOI*=-1

        if symbol in df_dict:
            #print(df_dict)
            df_dict[symbol]['Date'].append(current_str)
            df_dict[symbol]['volOI'].append(volOI)
            df_dict[symbol]['IV'].append(iv)
            df_dict[symbol]['delta'].append(delta)
        else:
            df_dict[symbol] = {'Date':[current_str],'volOI':[volOI],'IV':[iv],'delta':[delta]}

    current_date = current_date + timedelta(days=1)
    current_str = datetime.strftime(current_date,'%Y-%m-%d')

    for symbol,df in df_dict.items():
        try:
            with open(f'data_misc/optionsflow/{symbol}.dat','r') as f:
                d = f.read()
            data = pd.read_json(d,convert_axes=True)
            df = pd.concat([data,df],ignore_index=True)
            j = data.to_json(orient='records',date_format='iso')
        except:
            j = df.to_json(orient='records',date_format='iso')

        with open(f'data_misc/optionsflow/{symbol}.dat','w') as f:
            f.write(j)





def optionsflow_init(symbol:str,start_date=datetime(2016,1,1),end_date=datetime(2023,9,1)):
    if symbol=='BRK-B':
        symbol = 'BRK.B'
    with open(f'data_misc/optionsflow/{symbol}.dat','r') as f:
        d = f.read()
    df = pd.read_json(d,convert_axes=True)
    df = df[['Date','volOI']]
    df = fn.data_fill(df,damping_columns='all',start_date=start_date,end_date=end_date)
    return df



def get_pricetargets(symbol:str,start_date=datetime(2016,1,1),end_date=datetime(2023,6,1),time_interval:int=5,file_method='w'):
    if start_date is not None:
        current_date = start_date
        next_date = current_date + timedelta(days=time_interval)
        current_str = datetime.strftime(current_date,'%Y-%m-%d')
        next_str = datetime.strftime(next_date,'%Y-%m-%d')

    with open(f'data_equity/{symbol}.dat','r') as file:
        r = file.read()
    price_dat = pd.read_json(r,convert_axes=True)
    price_dat = price_dat[['Date','Adj Close']]
    price_dat.rename(columns={'Adj Close':'Close'},inplace=True)

    target_dat = {'Date':[],'Target':[],'GradeChange':[]}
    count = 0 #keep track of how many times the data is empty

    if start_date is not None:
        while current_date<=end_date:
            url = f'https://api.benzinga.com/api/v2.1/calendar/ratings?token=1c2735820e984715bc4081264135cb90&pagesize=1000&parameters[date_from]={current_str}&parameters[date_to]={next_str}&parameters[tickers]={symbol}&fields=fields%3Did,ticker,analyst_id,ratings_accuracy.smart_score,analyst_name,action_company,action_pt,adjusted_pt_current,adjusted_pt_prior,analyst,analyst_name,currency,date,exchange,id,importance,name,notes,pt_current,pt_prior,rating_current,rating_prior,ticker,time,updated,url,url_calendar,url_news,logo,quote'

            payload = {}
            headers = {
            'authority': 'api.benzinga.com',
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'en-US,en;q=0.9',
            'origin': 'https://www.benzinga.com',
            'referer': 'https://www.benzinga.com/',
            'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
            }

            response = requests.request("GET", url, headers=headers, data=payload)
            data = response.json()


            if data:
                count=0
                for item in data['ratings']:
                    date = item['date']
                    print(date)
                    try:
                        pt = float(item['adjusted_pt_current'])
                        pt_prior = float(item['adjusted_pt_prior'])
                    except:
                        continue

                    #date_dt = datetime.strptime(date,)
                    try:
                        close = price_dat.loc[price_dat['Date'] == date, 'Close'].values[0]
                    except:
                        print('Close price loc error')
                        continue
                    #print(f'Close: {close} \t PT: {pt}')
                    grade_change = (pt-pt_prior)/pt_prior

                    target_dat['Date'].append(date)
                    target_dat['Target'].append(pt-close)
                    target_dat['GradeChange'].append(grade_change)

            else:
                count+=1        
            
            current_date=next_date+timedelta(days=1)
            next_date = current_date + timedelta(days=time_interval)
            current_str = datetime.strftime(current_date,'%Y-%m-%d')
            next_str = datetime.strftime(next_date,'%Y-%m-%d')

            if count>=10:
                #terminate_and_run_proton(proton_path)
                count=0

    else:
        url=f'https://api.benzinga.com/api/v2.1/calendar/ratings?token=1c2735820e984715bc4081264135cb90&pagesize=1000&parameters[tickers]={symbol}&fields=fields%3Did,ticker,analyst_id,ratings_accuracy.smart_score,analyst_name,action_company,action_pt,adjusted_pt_current,adjusted_pt_prior,analyst,analyst_name,currency,date,exchange,id,importance,name,notes,pt_current,pt_prior,rating_current,rating_prior,ticker,time,updated,url,url_calendar,url_news,logo,quote'
        payload = {}
        headers = {
        'authority': 'api.benzinga.com',
        'accept': 'application/json, text/plain, */*',
        'accept-language': 'en-US,en;q=0.9',
        'origin': 'https://www.benzinga.com',
        'referer': 'https://www.benzinga.com/',
        'sec-ch-ua': '"Chromium";v="116", "Not)A;Brand";v="24", "Google Chrome";v="116"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'
        }
        response = requests.request("GET", url, headers=headers, data=payload)
        data = response.json()
        if data:
            for item in data['ratings']:
                date = item['date']
                #print(date)
                try:
                    pt = float(item['adjusted_pt_current'])
                    pt_prior = float(item['adjusted_pt_prior'])
                except:
                    continue

                #date_dt = datetime.strptime(date,)
                try:
                    close = price_dat.loc[price_dat['Date'] == date, 'Close'].values[0]
                except:
                    #print('Close price loc error')
                    continue
                #print(f'Close: {close} \t PT: {pt}')
                grade_change = (pt-pt_prior)/pt_prior
                print('made it through block')
                target_dat['Date'].append(date)
                target_dat['Target'].append(pt-close)
                target_dat['GradeChange'].append(grade_change)
        else:
            print('price target data error')


    df = pd.DataFrame(target_dat)
    if len(df) == 0:
        if start_date is None:
            start_date = datetime(2016,1,1)
        #if the pricetarget function fails to get data, fill it with blank data
        df = pd.DataFrame({'Date':[start_date],'Target':[0],'GradeChange':[0]})

    j = df.to_json(orient='records',date_format='iso')

    if file_method is not None:
        with open(f'data_misc/pricetargets/{symbol}.dat',file_method) as file:
            file.write(j)
            
    return df



def pricetargets_init(symbol:str='none',df=None,start_date=None,end_date=None):  
    if symbol!='none':
        with open(f'data_misc/pricetargets/{symbol}.dat','r') as file:
            r = file.read()
        data = pd.read_json(r)
    else:
        data = df
    
    if len(data) == 0:
        if start_date is None:
            start_date = datetime(2016,1,1)
        #if the pricetarget function fails to get data, fill it with blank data
        data = pd.DataFrame({'Date':[start_date],'Target':[0],'GradeChange':[0]})

    dataframe = fn.data_fill(data,start_date=start_date,end_date=end_date)
    return dataframe




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
                   pricetargets=True,
                   start_date=None,
                   end_date=None,
                   sequences=True,
                   sequence_length=16,
                   optionsflow=True,
                   target='1d',
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
            print(insider)

        if retail_sentiment:
            retail_df = fn.retail_sentiment_formatter(sym,start_date=start_date,end_date=end_date)
            data_list.append(retail_df)

        if pricetargets:
            pricetarget_df = pricetargets_init(sym,start_date=start_date,end_date=end_date)
            data_list.append(pricetarget_df)

        if optionsflow:
            optionsf_df = optionsflow_init(sym,start_date=start_date,end_date=end_date)
            data_list.append(optionsf_df)
            print(optionsf_df)
        
        for code in fed_list:
            data_list.append(fed_data[code])

        for city in city_list:
            data_list.append(housing_data[city])

        if target=='5d' or target=='all':
            data_list.append(close_5d_init(sym,start_date=start_date,end_date=end_date))

        if len(data_list)<2:
            df=data_list[0]
        else:
            df = fn.concatenate_data(data_list,start_date=start_date,end_date=end_date)
            #df.to_csv(f'csv_tests/concatenate_{sym}.csv')


        df = df.fillna(0)
        df.set_index('Date',inplace=True)
        if sequences:
            '''
            if pricetargets:
                df = fn.sequencizer(df,sequence_length,ignore_columns=['reportedEPS','surprise','sentiment','Close_5d','Target','GradeChange'])
            else:
                df = fn.sequencizer(df,sequence_length,ignore_columns=['reportedEPS','surprise','sentiment','Close_5d'])
            '''
            df = fn.sequencizer(df,sequence_length,ignore_columns=['Close','Close_5d'])
        if target=='1d' or target=='all': 
            df['Close_Tmr'] = df['Close'].shift(-1)
            df = df.drop(index=df.index[-1]) #drop the last row because the above shift method will result in NaN values

        if target!='1d' and target!='5d' and target != 'all':
            print('target error')
            return -1
        print(df)

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
                   update_earnings=True,
                   retail_sentiment=True,
                   google_trend=True,
                   insider_data=True,
                   pricetargets = True,
                   sequences = True,
                   optionsflow=True,
                   sequence_length = 16,
                   end_date = None
                   ): #hist is the number of days of history you want to include in the evaluation tensor
    if end_date is None:
        end_date = datetime.now()

    hist = sequence_length*3
    start_date = end_date - timedelta(days=hist)

    df_list = []
    processes = []
    data_sources = []
    '''
    def data_appender_worker(args):
        pth, get_function, symbol, start_date, end_date, concat, initialization_function = args
        data = data_appender(pth, get_function, symbol, start_date=start_date, end_date=end_date, concat=concat, overwrite=concat)
        if initialization_function.__name__ == news_init.__name__:
            initialized_data = initialization_function(symbol=symbol,df=None, start_date=start_date, end_date=end_date)
        else:
            initialized_data = initialization_function(symbol='none',df=data, start_date=start_date, end_date=end_date)
        return initialized_data
    '''
    #data_sources.append(equity_pth,get_equity_data,stock,start_date,end_date,True,fn.equity_formatter)
    #equity price
    equity_pth = f'data_equity/{stock}.dat'
    equity_df = data_appender(equity_pth,get_equity_data,stock,start_date=start_date,end_date=end_date,concat=True,overwrite=True)
    equity_df = fn.equity_formatter(df=equity_df,start_date=start_date,end_date=end_date)
    df_list.append(equity_df)
    #print(equity_df)

    #earnings
    if earnings:
        earnings_pth = f'data_equity/{stock}_earnings.dat'
        
        if not update_earnings:
            file_path = os.path.join(os.getcwd(), earnings_pth)
            if not os.path.exists(file_path):
                get_earnings(stock)
        else:
            earnings_df = data_appender(earnings_pth,get_earnings,stock,start_date=start_date,end_date=end_date,concat=True,overwrite=True)

        earnings_df = earnings_init(stock,start_date=start_date,end_date=end_date)
        df_list.append(earnings_df)
        #print('earnings')
        #print(earnings_df)
    
    if google_trend:
        trend_pth = f'data_misc/trend_{stock}.dat'
        trend_df = data_appender(trend_pth,get_search_trend_data,stock,start_date=start_date,end_date=end_date)
        trend_df = trend_init(symbol=stock,start_date=start_date,end_date=end_date)
        df_list.append(trend_df)
        #print('trend')
        #print(trend_df)

    #news sentiment
    def news_pth(outlet):
        pth = f'web_scraping/{stock}_{outlet}.dat'
        return pth

    if news_outlets:
        for outlet in news_outlets:
            pth = news_pth(outlet)
            if outlet == 'seekingalpha':
                data_appender(pth,get_seekingalpha_analysis,stock,start_date=start_date,end_date=end_date,concat=True,overwrite=True)
            elif outlet == 'bloomberg':
                try:
                    data_appender(pth,get_bloomberg_data,stock,start_date=start_date,end_date=end_date,concat=True,overwrite=True)
                except:
                    print(f'could not get {stock} bloomberg data')
            elif outlet == 'marketwatch':
                try:
                    data_appender(pth,get_marketwatch_data,stock,start_date=start_date,end_date=end_date,concat=True,overwrite=True)
                except:
                    print(f'could not get {stock} marketwatch data')

        news_sentiment_df = news_formatter(stock,outlets=news_outlets,start_date=start_date,end_date=end_date,file_method=None)
        #news_sentiment_df.drop(index=0,inplace=True)
        df_list.append(news_sentiment_df)
        #print(news_sentiment_df)

        #alpha_df = news_formatter(stock,outlets=['seekingalpha'],start_date=start_date,end_date=end_date,file_method=None)
        #df_list.append(alpha_df)
        #print(alpha_df)    

    if rsi:
        stock_rsi = fn.calculate_rsi(stock,start_date=start_date,end_date=end_date)
        df_list.append(stock_rsi)
        #print(stock_rsi)

    if insider_data:
        indsider_pth = f'data_misc/insider_{stock}.dat'
        try:
            insider_df = data_appender(indsider_pth,get_insider_trading_data,stock,start_date=start_date,end_date=end_date)
        except:
            get_insider_trading_data(stock,end_date=datetime.now())
            insider_df = data_appender(indsider_pth,get_insider_trading_data,stock,start_date=start_date,end_date=end_date)
        insider_df = insider_init(symbol=stock,start_date=start_date,end_date=end_date)
        df_list.append(insider_df)
        #print(insider_df)
    
    if retail_sentiment:
        retail_pth = f'data_equity/{stock}_retail_sentiment.dat'
        retail_df = data_appender(retail_pth,get_retail_sentiment,stock,start_date=start_date,end_date=end_date,concat=True,overwrite=True)
        retail_df = fn.retail_sentiment_formatter(stock,start_date=start_date,end_date=end_date)
        df_list.append(retail_df)
        #print(retail_df)

    if pricetargets:
        pricetarget_pth = f'data_misc/pricetargets/{stock}.dat'
        try:
            pricetarget_df = data_appender(pricetarget_pth,get_pricetargets,stock,start_date=start_date,end_date=end_date)
        except:
            get_pricetargets(stock,start_date=datetime(2021,1,1),end_date=datetime.now())
            pricetarget_df = data_appender(pricetarget_pth,get_pricetargets,stock,start_date=start_date,end_date=end_date)
        #pricetarget_df = pricetargets_init(symbol=stock,start_date=start_date-timedelta(days=30),end_date=end_date)
        pricetarget_df = pricetargets_init(symbol=stock,start_date=start_date,end_date=end_date)
        df_list.append(pricetarget_df)

    if optionsflow:
        #optionsflow_appender(end_date)
        optionsf_df = optionsflow_init(stock,start_date=start_date,end_date=end_date)
        df_list.append(optionsf_df)
    
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
            #print(fed_df)
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
            #print(city_df)
        print('Initialized housing data')

    df_merged = df_list[0]
    for df in df_list[1:]:
        df_merged = pd.merge(df_merged,df,on='Date',how='outer',)

    #df_merged['Close_Tmr'] = df_merged['Close'].shift(-1)
    if rsi:
        df_merged['RSI'].fillna(inplace=True,method='ffill')
    if retail_sentiment:
        df_merged['sentiment'].fillna(0,inplace=True)

    df_merged.set_index('Date',inplace=True)
    if sequences:
        df_merged = fn.sequencizer(df_merged,sequence_length,ignore_columns=['reportedEPS','surprise','sentiment','Target','GradeChange'])    

    date_str = datetime.strftime(end_date,'%Y-%m-%d')
    df_merged.to_csv(f'csv_data/equity/{stock}-{date_str}.csv')
    print(df_merged)
    return df_merged





#----------------------------------------------------------------------------
#EXECUTABLE CODE
#----------------------------------------------------------------------------

#sentiment_init(retail_stocks,retail_companies)
#equity_init(model_stocks,model_companies)
#as of now, seekingalpha_F.dat ends in 2015, so I am leaving ford out of the list

model_stocks = ['IBM','PEP','JWN','CVS','MU','CRM','LLY','UNH','GS','CAT','HD','CVX','NKE','KO','MA','MRK','AAPL','AAL','F','T','AXP','AMD',
                'BAC','BANC','UPS','DELL','DIS','INTC','GE','TGT','MCD','MSFT','NVDA','NFLX','QCOM','ROKU','RUN','SBUX','WMT',
                'GOOG','CSCO','CAT','MMM','PG','WBA','V','PFE','BRK-B','AMZN']

#for stock in model_stocks[2:]:
#    get_barrons_data(stock)

sequence_length=7

def get_all_data(symbol,start_date=datetime(2016,1,1),end_date=datetime.now(),marketwatch=True,bloomberg=True):
    #get_bloomberg_data(symbol)
    if marketwatch:
        get_marketwatch_data(symbol,start_date=start_date,end_date=end_date)
    if bloomberg:
        get_bloomberg_data(symbol)
    news_formatter(symbol,outlets=['bloomberg','marketwatch'],start_date=start_date,end_date=end_date)
    get_equity_data(symbol,start_date=start_date,end_date=end_date)
    get_earnings(symbol)
    get_insider_trading_data(symbol,max_page=50,start_date=start_date,end_date=end_date)
    get_pricetargets(symbol,start_date=start_date,end_date=end_date)
    get_retail_sentiment(symbol,start_date=start_date,end_date=end_date)


test_tickers = ['NYCB','TRV','EBAY','HTZ','VZ','WBD','SCHW','PYPL','PLUG','KEY','JPM']


training_data_init(test_tickers,
                   start_date=datetime(2020,5,18),
                   end_date=datetime(2023,9,1),
                   retail_sentiment=False,
                   google_trend=False,
                   earnings=False,
                   rsi=False,
                   pricetargets=False,
                   file_name='TEST-10-3',
                   sequence_length=sequence_length,
                   news_prefixes=[],
                   target='all'
                   )


successful_tickers = []
failed_tickers = []

end_date=datetime(2023,9,29)

#get_options_flow('AAPL',start_date=datetime(2020,8,1),end_date=datetime(2020,9,1))


def marketengine(sym_list=sec_data['ticker']):
    failed_tickers = []
    successful_tickers = []

    count = 0

    for sym in sym_list:
        sym = str(sym)
        try:
            eval_data_init(
                sym,
                google_trend=False,
                update_earnings=False,
                end_date=end_date, #IMPORTANT: delete this if you are running this during market hours or before midnight
                news_outlets=['bloomberg','marketwatch'],
                sequence_length=sequence_length
            )
            count = 0
            successful_tickers.append(sym)
        except Exception as e:
            print(f'Failed to get {sym} data: {e}')
            failed_tickers.append(sym)
            #count += 1
        if count>=5:
            print('FIVE SEQUENTIAL ERRORS, EXITING...')
            break

    print(f'Successful tickers: {successful_tickers}')
    print(f'Failed ticker: {failed_tickers}')

#marketengine()