import pandas as pd
import torch
import json
import requests
import quandl as qd
from datetime import datetime,timedelta

#ZILLOW HOUSING MARKET DATA
qd.ApiConfig.api_key = 'h3gFTxuuELMsc6zXU7_J'

nyMetro = qd.get_table('ZILLOW/DATA', indicator_id='ZSFH', region_id='394913') #NYC METROPOLITAN REGION ID
ny = "NewYork"
df_ny = pd.DataFrame(nyMetro)
json_ny = df_ny.to_json(orient='records',date_format='iso') #this line of code converts the pandas data frame to a json object oriented by rows
#parsed_ny = {"data":json.loads(json_ny)} #this line of code gets rid of the redundant backslashes (have to write a dummy dictionary because json.dump expects a dict input)
with open(f'data_misc/{ny}.dat','w') as file:
#    json.dump(parsed_ny,file)  
    file.write(json_ny)

laMetro = qd.get_table('ZILLOW/DATA', indicator_id='ZSFH',region_id='753899') #LOS ANGELES COUNTY AND ORANGE COUNTY ID
la = "LosAngeles"
df_la = pd.DataFrame(laMetro)
json_la = df_la.to_json(orient='records',date_format='iso')
#json_la.strip('[]')
#parsed_la = json.loads(json_la)
with open(f'data_misc/{la}.dat','w') as file:
    file.write(json_la)

chicagoMetro = qd.get_table('ZILLOW/DATA', indicator_id='ZSFH', region_id='394463') #CHICAGO METROPOLITAN REGION ID
chicago = "Chicago"
df_chicago = pd.DataFrame(chicagoMetro)
json_chicago = df_chicago.to_json(orient='records',date_format='iso')
#parsed_chicago = {"data":json.loads(json_chicago)}
with open(f'data_misc/{chicago}.dat','w') as file:
    file.write(json_chicago)
#ok this goes pretty far back. Fill in missing time series data with a forward fill.
#backward fill will be a little different, possible a simply k nearest neighbors algorithm

