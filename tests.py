import functions as fn
import pandas as pd

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
'''

#df = fn.housing_formatter('NewYork')
df = fn.equity_formatter('UPS',nominal=True)
print(df.head(30))