import functions as fn
import pandas as pd
from datetime import datetime

#df = fn.equity_formatter('DIS')
#print(df)
'''
df_list = []
def open_news_data(symbol,outlet):
    with open(f'web_scraping/{symbol}_{outlet}.dat','r') as file:
        r = file.read()
    df = pd.read_json(r)
    return df
df_list.append(open_news_data('AAL','bloomberg'))
df_list.append(open_news_data('AAL','marketwatch'))

count = 1
new_list = []
for df in df_list:
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    #df['Date'] = df['Date'].dt.set(hour=0, minute=0, second=0, microsecond=0)
    df['Date'] = pd.to_datetime(df['Date'])
    #df.rename(columns={'Headline':f'Headline_{count}'},inplace=True)
    df.drop(columns=['Summary'],inplace=True)
    df.reset_index(inplace=True,drop=True)
    count+=1
    new_list.append(df)

df = pd.concat(new_list,axis=0)
print(df)

def earnings_init(symbol:str):
    with open(f'data_equity/{symbol}_earnings.dat','r') as file:
        r = file.read()
    data = pd.read_json(r,convert_dates=True)
    df = fn.datetime_forward_fill(data)
    return df

def interpolate_dataframe(df):
    # Ensure the DataFrame is sorted by datetime
    df.loc[:,'datetime'] = pd.to_datetime(df['datetime'])
    df.sort_values(by='datetime', inplace=True)
    
    # Generate a date range with daily intervals between the min and max datetime
    date_range = pd.date_range(start=df['datetime'].min(), end=df['datetime'].max(), freq='D')
    
    # Reindex the DataFrame with the generated date range
    df = df.set_index('datetime').reindex(date_range)
    
    # Linearly interpolate the missing values
    df.interpolate(method='linear', inplace=True)
    df.index = df.index.set_names(['Date'])
    df.reset_index(inplace=True)
    
    return df

df = {'datetime':['2023-01-01','2023-01-01','2023-01-01','2023-01-07','2023-01-07','2023-01-11'],'values':[5,7,9,8,10,-2]}
df = pd.DataFrame(df)
df['datetime'] = pd.to_datetime(df['datetime'])
#df_new = df_new.sort_values(by='Date')
new_df = df.groupby('datetime').mean().reset_index()
new_df.loc[:, ['values']] = new_df[['values']].pct_change()
# Drop the first row with NaN values
new_df.drop(index=0, inplace=True)
#new_df = interpolate_dataframe(df)
print(new_df)


#df = fn.housing_formatter('NewYork')


df = {'Date':['2023-01-01','2023-01-07','2023-01-11'],'values':[5,7,9],'nums':[2,3,6],'pol':[8,20,30]}
df=pd.DataFrame(df)
df.loc[:,'Date'] = pd.to_datetime(df['Date'])
#interpolated = fn.linear_interpolate(df)
df_new = fn.data_fill(df,damping_columns=['values'],interpolate_columns=['pol'],end_date=datetime(2023,1,15))
print(df_new)
'''
'''
import numpy as np

def append_previous_n_data_points(arr, n, direction='left'):
    m, k = arr.shape
    p = n-1
    n = k*(n-1)
    new_k = k + n
    new_arr = np.zeros((m, n), dtype=arr.dtype)

    if direction == 'right':
        for i in range(m):
            new_arr[i, n:] = arr[i]
            if i > 0:
                new_arr[i, :n] = arr[i - 1, -n:]
        new_arr = new_arr[p:]
    elif direction == 'left':
        for i in range(m):
            new_arr[i, :k] = arr[i]
            if i < m - 1:
                new_arr[i, k:] = arr[i + 1]
        new_arr = new_arr[:-p]
    else:
        raise ValueError("Direction must be 'right' or 'left'.")

    return new_arr

array = [[1, 13, 6,5,1],[9, 4, 7,9,0],[19, 16, 2,-6,1],[2,3,5,-6,0],[7,7,17,-5,1]]
input_arr = np.array(array)

#append_arr = append_previous_n_data_points(input_arr,3)
#print(append_arr)


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

'''
def remove_weekend_rows(df:pd.DataFrame,average:bool=True): 
    #setting average = True averages the weekend column with the most recent data available
    # Check if the DataFrame has a 'Date' column
    if 'Date' not in df.columns:
        print("Error: 'Date' column not found in the DataFrame.")
        return None

    # Filter out weekend rows (Saturday and Sunday)
    #df = df[df['Date'].dt.weekday < 5]
    index = 0
    hit_weekday = False

    while index < df.shape[0]:
        if hit_weekday == False:
            if df['Date'].iloc[index].dt.weekday<5:
                hit_weekday=True
            else:
                df.drop(index,axis=0,inplace=True)
            index += 1
            continue
        
        if df['Date'].iloc[index].dt.weekday>4:
            for col in list(df.columns[1:]):
                val1 = float(df[col].iloc[index])
                val2 = float(df[col].iloc[index-1]) 
                avg = (val1+val2)/2
                df[col].iloc[index-1] = avg
            df.drop(index,axis=0,inplace=True)
        index+=1
    
    #return averaged_df
'''
import re
df = pd.read_csv('csv_tests/comparison_classifier-9-2.csv')
threshold = 0.60

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