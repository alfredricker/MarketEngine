import pandas as pd
import torch
import json
import requests
import quandl as qd
from datetime import datetime,timedelta
import functions

#ZILLOW HOUSING MARKET DATA
qd.ApiConfig.api_key = 'h3gFTxuuELMsc6zXU7_J'

#see data_misc/city_realestate_data.csv for the region_ids of a given city 
def fetch_housing_data(city,id):
    metro = qd.get_table('ZILLOW/DATA', indicator_id='ZSFH', region_id=id)
    df = pd.DataFrame(metro)
    j = df.to_json(orient='records',date_format='iso')
    j.strip('[]')
    with open(f'data_misc/{city}.dat','w') as file:
        file.write(j)
    return 0

#commenting some of this section out because the data has already been fetched
ny = "NewYork"
#fetch_housing_data(ny,'394913')

la = "LosAngeles"
#fetch_housing_data(la,'753899')

chicago = "Chicago"
#fetch_housing_data(chicago,'394463')

#ok this goes pretty far back. Fill in missing time series data with a forward fill.
#backward fill will be a little different, possible a simply k nearest neighbors algorithm


#this section formats the data properly
def rewrite_housing_data(city):
    h = functions.housing_formatter(city)
    h_json = h.to_json(orient='records',date_format='iso')
    h_json.strip('[]')
    with open(f'data_misc/{city}_formatted.dat','w') as f: #i want to keep the old data in case I need it so I'm writing a new file with "formatted" indicator
        f.write(h_json)
    return 0

rewrite_housing_data(ny)
rewrite_housing_data(la)
rewrite_housing_data(chicago)
