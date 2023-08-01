import pandas as pd
import torch
import json
import requests
import quandl
from datetime import datetime,timedelta

quandl.ApiConfig.api_key = 'h3gFTxuuELMsc6zXU7_J'

def form_two_columns(row):
    row_str = str(row)
    row_list = row_str.split('\n',1)
    value = float(row_list[0].split(' ',1)[1])
    date = (row_list[1].split(' ',2)[1])
    return [date,value]

'''
#HISTORICAL GDP
gdp = "GDP_US"
gdp_get = quandl.get('FRED/GDP')
gdp_df = pd.DataFrame(gdp_get)
new_df = {'Date':[],'Value':[]}
# Loop through each row of the dataframe and apply the form_two_columns function
for index, value_row in gdp_df.iterrows():  
    arr = form_two_columns(value_row)
    new_df['Date'].append(arr[0])
    new_df['Value'].append(arr[1])
# Create a new DataFrame from the processed_rows list with appropriate column names
gdp_df = pd.DataFrame(new_df)
# Convert the 'Date' column to a datetime type
gdp_df['Date'] = pd.to_datetime(gdp_df['Date'])
gdp_json = gdp_df.to_json(orient='records',date_format='iso')
gdp_json.strip('[]')
gdp_load = json.loads(gdp_json)
with open(f'data_fed/{gdp}.dat','w') as file:
    #file.write(gdp_json)
    json.dump(gdp_load,file)
'''

#CPI for all urban consumer items
cpi = "CPI_US"
cpi_get = quandl.get('FRED/CPIAUCSL')
cpi_df = pd.DataFrame(cpi_get)
new_df = {'Date':[],'Value':[]}
for index,value_row in cpi_df.iterrows():
    arr = form_two_columns(value_row)
    new_df['Date'].append(arr[0])
    new_df['Value'].append(arr[1])
cpi_df = pd.DataFrame(new_df)
cpi_df['Date'] = pd.to_datetime(cpi_df['Date'])
cpi_json = cpi_df.to_json(orient='records',date_format='iso')
cpi_json.strip('[]')
cpi_load = json.loads(cpi_json)
with open(f'data_fed/{cpi}.dat','w') as file:
    json.dump(cpi_load,file)


#M1 Money stock and velocity of M1 money supply (M1V)
cpi = "CPI_US"
m1_get = quandl.get('FRED/M1')
m1_df = pd.DataFrame(m1_get)
m1v_get = quandl.get('FRED/M1V')
m1v_df = pd.DataFrame(m1v_get)
new_df = {'Date':[],'M1':[]}
for index,value_row in m1_df.iterrows():
    arr = form_two_columns(value_row)
    new_df['Date'].append(arr[0])
    new_df['M1'].append(arr[1])
m1_df = pd.DataFrame(new_df)
m1_df['Date'] = pd.to_datetime(m1_df['Date'])
print(m1_df)
m1_json = m1_df.to_json(orient='records',date_format='iso')
m1_json.strip('[]')
m1_load = json.loads(cpi_json)
with open(f'data_fed/m1.dat','w') as file:
    json.dump(m1_load,file)
