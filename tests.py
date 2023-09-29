import functions as fn
import pandas as pd
from datetime import datetime,timedelta
import numpy as np
#from data_compilation import get_pricetargets

#df = fn.equity_formatter('DIS')
#print(df)



'''


df = {'Date':['2023-01-01','2023-01-07','2023-01-11'],'values':[5,7,9],'nums':[2,3,6],'pol':[8,20,30]}
df=pd.DataFrame(df)
df.loc[:,'Date'] = pd.to_datetime(df['Date'])
#interpolated = fn.linear_interpolate(df)
df_new = fn.data_fill(df,damping_columns=['values'],interpolate_columns=['pol'],end_date=datetime(2023,1,15))
print(df_new)
'''
'''
import numpy as np




def sequencizer(X,n,direction='left'):
    m,k = X.shape
    k=k-1
    p = k*n
    new_arr = np.zeros((m,p),dtype=X.dtype)

    if direction == 'left':
        for i in range(m - n + 1):
            new_arr[i, :] = X[i:i + n].ravel()
        new_arr = new_arr[:-(n-1)]
    return new_arr

#sequences = sequencizer(input_arr,2)
#print(sequences)

def sequencizer_df(X, n, direction='left'):
    m, k = X.shape
    p = k * n
    new_columns = []

    if direction == 'left':
        for i in range(m - n + 1):
            for j in range(n):
                new_columns.extend([f"{col}_prev{j+1}" for col in X.columns])
                X = X.reset_index(drop=True)
                new_row = X.iloc[i:i + n].values.ravel()
                X = X.drop(columns=list(X.columns))
                X = pd.DataFrame(new_row.reshape(1, -1), columns=new_columns)
    else:
        raise ValueError("Direction must be 'left'.")

    return X

# Example usage:
data = {'A': [1, 9, 19, 2, 7],
        'B': [13, 4, 16, 3, 7],
        'C': [6, 7, 2, 5, 17]}

df = pd.DataFrame(data)
n = 3
direction = 'left'

result_df = sequencizer_df(df, n, direction)
print(result_df)
'''
'''
data = {'ticker': ['AAL', 'AAPL', 'AMD', 'AMZN', 'BAC', 'BANC', 'BRK-B', 'CGNX', 'CSCO', 'DELL', 'DIS', 'F', 'GE', 'GOOG', 'INTC', 'MCD', 'META', 'MLM', 'MSFT', 'NFLX', 'NVDA', 'QCOM', 'ROKU', 'RUN', 'SBUX', 'SHOP', 'T', 'TGT', 'TSLA', 'UPS', 'WMT'],
        'cik': [6201, 320193, 2488, 1018724, 70858, 1169770, 1067983, 851205, 858877, 1571996, 1744489, 37996, 40545, 1652044, 50863, 63908, 1326801, 916076, 789019, 1065280, 1045810, 804328, 1428439, 1469367, 829224, 1594805, 732717, 27419, 1318605, 1090727, 104169]}

df = pd.DataFrame(data)
df.to_csv('csv_data/SEC_CODES.csv')
'''




import re
df = pd.read_csv('csv_tests/comparison_classifier-9-28.csv')
threshold = 0.75

def get_probs(tensor):
    tensor = tensor.strip('()')
    split = tensor.split(',')
    prob_0 = re.search(r'\d+\.\d+', split[0])
    prob_1 = re.search(r'\d+\.\d+', split[1])
    return (float(prob_0.group()),float(prob_1.group()))

count_0 = 0
correct_0 = 0
count_1 = 0
correct_1 = 0

for index,value in df.iterrows():
    actual = int(value['Actual'])
    predicted = int(value['Predicted'])
    probs = get_probs(value['Predicted-Probs'])

    if probs[0]>threshold:
        count_0+=1
        if actual == 0:
            correct_0+=1
    
    if probs[1]>threshold:
        count_1+=1
        if actual==1:
            correct_1+=1
print(f'count 0: {count_0}')
print(f'accuracy 0: {correct_0/count_0}')
print(f'count 1: {count_1}')
print(f'accuracy 1: {correct_1/count_1}')

sym_list = ['AAPL','AMD','AMZN','BAC','BANC','BRK-B','UPS','DELL','DIS','MRVL', 'WBD','CSX', 'PYPL', 'JNJ', 'RBLX','BAX',
            'ET', 'NOV', 'DISH', 'LUMN', 'KO', 'SPWR', 'PEP', 'LAZR', 'MAT','HOOD', 'AMAT', 'WBA', 'EYE', 'JPM', 'FHN', 'CARR',
            'INTC','GE','TGT','MCD','MSFT','NVDA','NFLX','QCOM','ROKU','WMT','GOOG','CSCO','AXP','CAT','MMM','PG','WBA',
            'V','AKAM','AES','BYND','BAC','CMCSA','FITB','FLEX','IP','LSXMK','MTCH','PLUG','RKT','T','UBER','VICI','VTRS',
            'SMPL','VZ','DAL','HAL','PFE','ET','HLIT','FTAI','UCTT','XPOF','NOG','A','AA','ABBV','ABNB','ABT',
            'ACHR','ADM','LCID','SOUN','QS','LUV','AI','ENVX','NYCB','MU','NCLH','KEY','BMY','OPEN','LYFT','KDP',
            'RIOT','CVNA','UPST','NOVA','FCEL','SBUX','RUN','FUBO', 'RCL', 'PCG', 'EBAY', 'GM', 'SPCE',
            'JBLU', 'BA', 'GPS', 'CROX', 'IONQ', 'WU', 'ARR', 'MXL','COIN', 'SNAP', 'HBAN', 'AR', 'FSR',
            'FCX', 'KMI','SCHW', 'PARA', 'PINS', 'PTEN','RTX','TDOC','WSC','USB']




'''
import time
import yfinance as yf
import json
import requests

start_time = time.time()

# Your code to measure runtime goes here

import os



end_time = time.time()
runtime = end_time - start_time

print(f"Runtime: {runtime:.4f} seconds")
'''

