import pandas as pd
import torch
import json
import requests
import quandl
from datetime import datetime,timedelta
from functions import form_two_columns,fed_formatter

quandl.ApiConfig.api_key = 'h3gFTxuuELMsc6zXU7_J'

#see https://data.nasdaq.com/data/FRED-federal-reserve-economic-data/documentation
def get_fred_data(code):
    code_get = quandl.get(f'FRED/{code}')
    code_df = pd.DataFrame(code_get)
    new_df = {'Date':[],'Value':[]}
    for index,value_row in code_df.iterrows():
        arr = form_two_columns(value_row)
        new_df['Date'].append(arr[0])
        new_df['Value'].append(arr[1])
    code_df = pd.DataFrame(new_df)
    code_df['Date'] = pd.to_datetime(code_df['Date'])
    code_json = code_df.to_json(orient='records',date_format='iso')
    code_json.strip('[]')
    code_load = json.loads(code_json)
    with open(f'data_fed/{code}.dat','w') as file:
        json.dump(code_load,file)
    print(f'Got {code} FED data')
    return code_load

'''
gdp = get_fred_data('GDP')
cpi = get_fred_data('CPIAUCSL')
m1 = get_fred_data('M1')
m1v = get_fred_data('M1V')
fedfunds = get_fred_data('DFF')
threemo_tbill = get_fred_data('DTB3')
unemployment = get_fred_data('UNRATE')
industrial_prod = get_fred_data('INDPRO')
'''

def rewrite_fed_data(code,nominal):
    formatt = fed_formatter(code,nominal)
    j = formatt.to_json(orient='records',date_format='iso')
    j.strip('[]')
    with open(f'data_fed/{code}_formatted.dat','w') as file:
        file.write(j)
    return 0

rewrite_fed_data('GDP',nominal=False)
rewrite_fed_data('CPIAUCSL',nominal=False)
rewrite_fed_data('M1',nominal=False)
rewrite_fed_data('M1V',nominal=True)
rewrite_fed_data('DFF',nominal=True)
rewrite_fed_data('DTB3',nominal=True)
rewrite_fed_data('UNRATE',nominal=True)
rewrite_fed_data('INDPRO',nominal=False)

