import pandas as pd
import torch
import json
import requests
from datetime import datetime,timedelta

#ZILLOW HOUSING MARKET DATA
#use the ZHVI (home value index) indicator. Another thing to test the effect of is the home value forecast (ZHVF)
#this is the Nasdaq Data Link zillow data.
'''
url = "https://data.nasdaq.com/api/v3/datatables/ZILLOW/DATA?indicator_id=ZATT&region_id=270957&api_key=h3gFTxuuELMsc6zXU7_J"
try:
    r = requests.get(url)
    data = r.json()
    if 'Error' in data:
        print("Error in Zillow data request")
    else:
        with open(f'data_misc/zillow.dat','w') as file:
            json.dump(data,file)
            
except:
    print("An error occured: could not request Zillow data")
'''
#this data here is probably better, it is from bridge interactive zestimates

url = "https://api.bridgedataoutput.com/api/v2/zestimates_v2/zestimates?access_token=h3gFTxuuELMsc6zXU7_J"
try:
    req = requests.get(url)
    zestimate = req.json()
    if 'error' in zestimate:
        print("Error in Zestimate request")
    else:
        with open(f'data_misc/zestimate.dat','w') as file:
            json.dump(zestimate,file)
except:
    print("An error occured: could not request Zestimate data")