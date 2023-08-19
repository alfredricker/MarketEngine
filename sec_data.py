import requests
import json
from bs4 import BeautifulSoup
import pandas as pd

#IMPORTANT: the sec locates company urls by the "CIK" number, which is a unique identifier for each company. I'm going to have to include them in a dictionary.
cik = {'AAL':6201,'AAPL':320193,'AMD':2488,
       'AMZN':1018724,'BAC':70858,'BRK-B':1067983,'CGNX':851205,
       'DELL':1571996,'DIS':1744489,'F':37996,'GE':40545,
       'GOOG':1652044,'INTC':50863,'MCD':63908,'META':1326801,'NFLX':1065280,
       'NVDA':1045810,'ROKU':1428439,'SBUX':829224,'SHOP':1594805,
       'T':732717,'TGT':27419,'TSLA':1318605,'WMT':104169}

def get_insider_trading_data(symbol:str,max_page:int=6):
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

            f_in_kind = data[5].text.strip()

            first_number = float(data[7].text.strip())
            second_number = float(data[8].text.strip())
            if second_number == 0:
                continue #I'll just skip this iteration if the shares owned is zero because I'm not sure what to replace it with

            change = first_number/second_number
            if buy_or_sell == 'D':
                change *= -1
            df['Date'].append(date)
            df['InsiderFlow'].append(change)
    
    df = pd.DataFrame(df)
    j = df.to_json(orient='records',date_format='iso')
    with open(f'data_misc/insider_{symbol}.dat','w') as file:
        file.write(j)


get_insider_trading_data('AAL')