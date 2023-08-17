import functions as fn
import pandas as pd

#df = fn.equity_formatter('DIS')
#print(df)

df_list = []
def open_news_data(symbol,outlet):
    with open(f'web_scraping/{symbol}_{outlet}.dat','r') as file:
        r = file.read()
    df = pd.read_json(r)
    return df
df_list.append(open_news_data('BAC','bloomberg'))
df_list.append(open_news_data('BAC','marketwatch'))

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
