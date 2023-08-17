#THIS FILE COMPILES ALL OF THE RELEVANT EQUITY DATA THAT YOU WOULD LIKE TO PROCESS INTO ONE DATAFRAME THEN EXPORTS IT TO data.csv
#This file also compiles all of the sentiment data and exports it to SENTIMENT.csv
import pandas as pd
import json
import functions as fn
from datetime import datetime

#MODEL STOCKS
#stocks currently in the model. It is import that these two lists are aligned ticker:company
model_stocks = ['AAPL','AMD','TSLA','F','GOOG','TUP','MSFT','NVDA','T','AMZN','META']
model_companies = ['Apple','AMD','Tesla','Ford','Google','TUP','Microsoft','Nvidia','AT&T','Amazon','Facebook']
retail_stocks = ['AAPL','AMD','AMZN','MSFT','NVDA','TSLA']
retail_companies = ['Apple','AMD','Amazon','Microsoft','Nvidia','Tesla']
#TREND DATA
def trend_init(company:str):
    with open(f'data_misc/trend_{company}.dat', 'r') as file:
        data = file.read()
    trend_df = pd.read_json(data)
    return trend_df

#FED DATA
def fed_init(code:str):
    with open(f'data_fed/{code}_formatted.dat','r') as file:
        r = file.read()
    df = pd.read_json(r)
    return df

gdp = 'gdp'
cpi = 'cpiaucsl'
dff = 'dff'
m1 = 'm1'
m1v = 'm1v'
indpro = 'indpro'
unrate = 'unrate'
gdp = fed_init(gdp)
cpi = fed_init(cpi)
dff = fed_init(indpro)
m1 = fed_init(m1)
m1v = fed_init(m1v)
indpro = fed_init(indpro)
unrate = fed_init(unrate)
print("Initialized fed data")

#HOUSING DATA
def housing_init(city:str):
    with open(f'data_misc/{city}_formatted.dat','r') as file:
        r = file.read()
    data = pd.read_json(r)
    return data
ny = housing_init('NewYork')
la = housing_init('LosAngeles')
chicago = housing_init('Chicago')
print("Initialized housing data")

#EQUITY/TARGET DATA
#surprise column contains a NaN value which interrupts the model as of now.
def equity_init(stocks_list,companies_list):
    df_list = [] #list of data frames that will be concatenated by row at the end of the function

    for i in range(len(stocks_list)):    
        stock_sym = stocks_list[i]
        print(f'Initializing {stocks_list[i]} data...')
        stock = fn.equity_formatter(stock_sym)
        stock_close = stock[0]
        stock_volume = stock[1]
        stock_rsi = fn.calculate_rsi(stock_sym)
        
        stock_trend = trend_init(companies_list[i])
        stock_earnings = fn.earnings_formatter(stock_sym)
        stock_reported_eps = stock_earnings[0]
        stock_surprise_eps = stock_earnings[1]

        #CONCATENATE ALL DATA INTO A SINGLE LARGE DATAFRAME
        data_list = [stock_close,stock_volume,stock_trend,stock_rsi,stock_reported_eps,stock_surprise_eps,gdp,cpi,dff,m1,unrate,ny]
        df = fn.concatenate_data(data_list)
        df['Close_Tmr'] = df['Close'].shift(-1)
        df = df.drop(index=df.index[-1]) #drop the last row because the above shift method will result in NaN values
        df.set_index('Date',inplace=True)
        df_list.append(df)

    data = pd.concat(df_list,axis=0,ignore_index=True)
    data.to_csv('DATA.csv')
    print(f'Initialized equity data')
    return data
#IMPORTANT: since I want to look at how the present day data effects the closing price of tomorrow, I have to copy the close column, and shift it up 1 position in a new row


#HUGGING FACE PRE TRAINED SENTIMENT MODEL
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


def roberta_sentiment(sentence:str):
    encoded_text = tokenizer(sentence,return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy() #returns sentiment scores as a np array of the form [neg,neutral,pos]
    scores = softmax(scores)
    #lets return one number in the range (-1,1), -1 being as negative as possible and 1 being as positive as possible
    sentiment = scores[0]*(-1.) + scores[1]*0. + scores[2]*1.
    return sentiment


def news_roberta(company,outlet): #pretty much the same as bloomberg init
    with open(f'web_scraping/{company}_{outlet}.dat','r') as file:
        r = file.read()
    data = pd.read_json(r)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values(by='Date', axis=0)
    sentiment_df = {'Headline_score':[]}
    for index in data.index:
        headline_score = roberta_sentiment(data['Headline'].iloc[index])
        #summary_score = roberta_sentiment(data['Summary'].iloc[index])
        sentiment_df['Headline_score'].append(headline_score)
        #sentiment_df['Summary_score'].append(summary_score)
    sentiment_df = pd.DataFrame(sentiment_df)
    #print(sentiment_df)
    df = pd.concat([data,sentiment_df],axis=1)
    #print(df)
    df = df[['Date','Headline_score']]
    #df.reset_index(inplace=True)
    return df


#the news_init function takes a while so I'm going to initialize one at a time and save them to dat files.
def news_formatter(company,symbol,outlets=['bloomberg','marketwatch']):
    df_list = []
    for outlet in outlets:
        df_list.append(news_roberta(symbol,outlet))
    if len(df_list)>1:
        new_list = []
        for df in df_list:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date')
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            df['Date'] = pd.to_datetime(df['Date'])
            df.drop(columns=['Summary'],inplace=True) #As of now I'm not using the summary column
            df.reset_index(inplace=True,drop=True)
            new_list.append(df)
        df = pd.concat(new_list,axis=0)
    else:
        df = df_list[0]
    #print(df)
    #collapse all the headline scores into one averaged column
    #average_series = df.apply(lambda row: row[1:].mean(), axis=1)
    #print(average_series)
    # Convert the average Series back to a DataFrame
    #average_df = pd.concat([df.iloc[:,0],pd.DataFrame({'Headline': average_series})],axis=1)
    df = fn.multiple_date_fill(df)
    j = df.to_json(orient='records',date_format='iso')
    with open(f'data_misc/news_sentiment_{symbol}.dat','w') as file:
        file.write(j)
    print(f'Successfully formatted {symbol} news data')
    return df


def news_init(symbol):
    with open(f'data_misc/news_sentiment_{symbol}.dat','r') as file:
        r = file.read()
    df = pd.read_json(r)
    return df


def sentiment_init(stocks_list,companies_list):
    df_list = []    
    for i in range(len(stocks_list)):
        retail_df = fn.retail_sentiment_formatter(stocks_list[i])
        news_sentiment = news_init(stocks_list[i])
        stock_trend = trend_init(companies_list[i])

        stock = fn.equity_formatter(stocks_list[i])
        stock_close = stock[0]
        stock_rsi = fn.calculate_rsi(stocks_list[i])

        data_list = [stock_close,stock_trend,retail_df,news_sentiment,stock_rsi]
        df = fn.concatenate_data(data_list)
        df['Close_Tmr'] = df['Close'].shift(-1)
        df = df.drop(index=df.index[-1]) #drop the last row because the above shift method will result in NaN values
        df.set_index('Date',inplace=True)
        df_list.append(df)

    data = pd.concat(df_list,axis=0,ignore_index=True)
    data.to_csv('DATA_SENTIMENT.csv')
    print(f'Initialized sentiment data')
    return data


#sentiment_init(retail_stocks,retail_companies)
#equity_init(model_stocks,model_companies)
news_formatter('ford','F',outlets=['marketwatch'])
