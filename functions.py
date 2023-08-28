from datetime import datetime,timedelta
from typing import List
import json
import pandas as pd
import numpy as np
import re

#this condition will be checked within several functions... nevermind I was recently made aware of df.fillna()
def contains_non_numeric(input_string):
    pattern = r'^[0-9]*\.?[0-9]+$'
    return not re.match(pattern, input_string)


def get_recent_trading_day():
    # Get the current date
    current_date = datetime.now()
    # Keep subtracting one day until we find a trading day (exclude weekends and holidays)
    while True:
        if current_date.weekday() >= 5:  # 5 and 6 correspond to Saturday and Sunday
            current_date -= timedelta(days=1)
        else:
            trading_day = current_date
            break
    return trading_day


#this function is necessary if the date column of the pd dataframe imports improperly
def form_two_columns(row):
    row_str = str(row)
    row_list = row_str.split('\n',1)
    value = float(row_list[0].split(' ',1)[1])
    date = (row_list[1].split(' ',2)[1])
    return [date,value]


#this function returns the min and max dates in a file containing json data
def get_json_date_extrema(file_path):
    with open(file_path,'r') as file:
        json_string = file.read()
    df = pd.read_json(json_string)
    start_date = min(df['date'])
    end_date = max(df['date'])
    arr = [start_date,end_date]
    return arr

#--------------------------------------------------------------------------------------------------------

def inverse_sqrt_damping(value:float,t:int,c:float): #c is damping constant, t is number of days since initial information
    y = value/np.sqrt(1+(t*t/(c*c)))
    return y



def ffill(df:pd.DataFrame,end_date:datetime=None,na=False,damping=False,damping_constant:float=0.0): #standard forward fill of the last value of the dataframe up to a specified date
    #the dataframe must already be in chronological order
    if df.columns[0]!='Date':
        print('fn.ffill() error: no Date in columns')
        return -1
    if na==True and damping==True:
        print('fn.ffill() error: both damping and na cannot be true')
        return -1
    if end_date is not None:
        if end_date<df['Date'].max():
            print('fn.ffill() error: end_date is less than max date, just use default:None')
        elif end_date==df['Date'].max():
            end_date=None
    
    new_df = pd.DataFrame(columns=df.columns)

    for i in range(len(df)):
        current_row = df.iloc[i]
        current_date = current_row['Date']
        date_range = []

        if i < len(df)-1:
            next_row = df.iloc[i+1]
            date_range = pd.date_range(current_date,next_row['Date'],inclusive='left')
        else:
            if end_date is None:
                break
            date_range = pd.date_range(current_date,end_date,inclusive='both')
        
        curr_values = [current_row[x] for x in df.columns[1:]]
        na_values = [pd.NA for x in df.columns[1:]]
        for date in date_range:
            if damping:
                t = (date-current_date).days
                values = [inverse_sqrt_damping(float(current_row[x]),t,damping_constant) for x in df.columns[1:]]
            elif na and date>current_date:
                values = na_values
            else:
                values = curr_values

            new_row = [date]+values
            new_df.loc[len(new_df)] = new_row

    if end_date is None:
        new_df.loc[len(new_df)] = df.iloc[-1] #need this line because of the non-inclusive loop

    return new_df



def data_fill(df:pd.DataFrame,
                 target_columns=None, #columns that you would like to keep in the dataframe (default None keeps all columns)
                 percent_columns=None, #fill with percent change rather than nominal values
                 damping_columns=None, #include a damping function to damp missing values towards zero as time->inf
                 damping_constant:float=15.0, #this determines the convexity of the damping
                 interpolate_columns=None, #linearly interpolate the values rather than fill in with most recent value
                 end_date=None): #the date to fill the data up to. default is the max date in the df
    #this function takes in a pandas data frame and fills in all missing daily values by taking the most recent value
    if 'Date' != df.columns[0]:
        return print("Error: 'Date' must be 0th column in dataframe")
    
    if target_columns is not None:
        target_list = target_columns.insert(0,'Date')
        df = df[target_list]

    columns = list(df.columns)
    if percent_columns=='all':
        percent_columns = df.columns[1:] #don't want to remove these columns from the list because we still need to fill in this data
    elif damping_columns=='all':
        damping_columns = df.columns[1:]

    df.loc[:,'Date'] = pd.to_datetime(df['Date'])
    df_sorted = df.sort_values(by='Date')
    df_sorted = df_sorted.groupby('Date').mean().reset_index()

    if percent_columns is not None:
        df_sorted.loc[:, percent_columns] = df_sorted[percent_columns].pct_change()
        # Drop the first row with NaN values
        df_sorted.drop(index=0, inplace=True)

    if damping_columns is not None:
        for col in damping_columns:
            columns.remove(col)

    if interpolate_columns is not None:
        for col in interpolate_columns:
            columns.remove(col)

    max_date = df_sorted.max(axis=0)[0]
    if end_date is None:
        end_date = max_date
    if end_date<max_date:
        print('forward_fill error: end_date must be >= max_date')
        return -1
    
    df_list = []
    if columns:
        df_std = df_sorted[columns]
        df_list.append(ffill(df_std,end_date=end_date))

    if interpolate_columns is not None:
        interpolate_columns.insert(0,'Date')
        df_inter = df_sorted[interpolate_columns]
        df_inter = ffill(df_inter,end_date=end_date,na=True)
        df_inter.interpolate(method='linear',inplace=True)
        df_list.append(df_inter)

    if damping_columns is not None:
        damping_columns.insert(0,'Date')
        df_damp = df_sorted[damping_columns]
        df_damp = ffill(df_damp,end_date=end_date,damping=True,damping_constant=damping_constant)
        df_list.append(df_damp)
    
    df_merged = df_list[0]  # Initialize with the first DataFrame
    for df in df_list[1:]:
        df_merged = pd.merge(df_merged, df, on='Date', how='outer')

    return df_merged    

#---------------------------------------------------------------------------------------------------


#this function makes all your dataframe timelines line up. input a list of pandas data frames
def get_df_date_extrema(df_list):
    min_dates = []
    max_dates = []
    for df in df_list:
        df['Date'] = pd.to_datetime(df['Date'])
        minimum = df['Date'].min(axis=0,skipna=True)#[0]
        min_dates.append(minimum)
        maximum = df['Date'].max(axis=0,skipna=True)
        #print(f'min_date:{min}, max_date:{max}')
        max_dates.append(maximum)
    #now get the largest min dates and the smallest max dates
    min_date = min(min_dates)
    max_date = max(max_dates)
    #print(f'[{min_date},{max_date}]')
    return [min_date,max_date]


#properly reads and formats the pd dataframe, forward fills the data, and splits data into a [price,volume] data list
def equity_formatter(symbol,nominal=False,api='yfinance'):
    with open(f'data_equity/{symbol}.dat','r') as file: #open the necessary equity file
        d = file.read()

    if api == 'yfinance':
        df = pd.read_json(d)
        df = df[['Date','Adj Close','Volume']]
        df = df.rename(columns={'Adj Close':'Close'})
    elif api=='alphavantage':
        j = json.loads(d)
        dat = j['Time Series (Daily)'] #get rid of the MetaData classifier in the json file
        df = pd.DataFrame(dat).T #create a dataframe and transpose it
        df.reset_index(inplace=True) #this line makes it so that the date becomes an accessible column rather than the index
        column_names = ['Date','Open','High','Low','Close','Volume']
        df.columns = column_names #rename the columns
        df = df[['Date','Close','Volume']] #I just want close for the first trial of this. I will come back and do stuff with high and low data as well
    else:
        print("Equity formatter error: supported APIs: yfinance, alphavantage")

    if nominal == False:  
        df = data_fill(df,percent_columns=['Close'],interpolate_columns=['Close'])
    elif nominal == True:
        df = data_fill(df,interpolate_columns=['Close'])
    else:
        print("Error in functions.py: Equity formatter error!")
    #print(f'Successfully formatted {symbol} data')
    return df




def retail_sentiment_formatter(symbol):
    with open(f'data_equity/{symbol}_retail_sentiment.dat','r') as file:
        j = file.read()
    df = pd.read_json(j)
    df.reset_index(inplace=True)
    df = df.fillna(0) #don't want to do forward fills
    df = df[['Date','sentiment']]
    return df


#returns the (almost) proper df to start the neural network process
def concatenate_data(df_list,start_date=None,end_date=None):
    min_max = get_df_date_extrema(df_list)
    if start_date is None:
        start_date = pd.to_datetime(min_max[0])
    if end_date is None:
        end_date = pd.to_datetime(min_max[1])

    df_filtered = []
    for df in df_list:
        df.loc[:,'Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')
        df = df[df['Date']>=start_date]
        df = df[df['Date']<=end_date]
        #print(df.shape[0]) #they are all the same shape
        df_filtered.append(df)
    count = 0
    for df in df_filtered:
        if count==0:
            result_df = df.copy()
            result_df = result_df.reset_index(drop=True)
            count+=1
        else:
            df = df.reset_index(drop=True)
            df_new = df.iloc[:,1:]
            result_df = pd.concat([result_df,df_new],axis=1)
    return result_df


#make it so that the target column has an equal number of ones and zeros. Note that the transform_to_binary function must be ran first
def remove_imbalances(df,target_column:str,sequence=False):
    if sequence==False: 
            # Check if the target column exists in the DataFrame
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

        # Separate the DataFrame into positive and negative samples
        positive_samples = df[df[target_column] == 1]
        negative_samples = df[df[target_column] == 0]
        # Determine the minimum number of samples for balancing
        min_samples = min(len(positive_samples), len(negative_samples))
        # Sample an equal number of positive and negative samples
        balanced_positive_samples = positive_samples.sample(min_samples)#, random_state=42)
        balanced_negative_samples = negative_samples.sample(min_samples)#, random_state=42)

        # Combine the balanced samples back into a single DataFrame
        balanced_df = pd.concat([balanced_positive_samples, balanced_negative_samples], ignore_index=True)
        return balanced_df
    else:
        #find the data frame in the sequences list of the smallest size and then set all other dataframes to be that size
        min_size = min(len(frame[0]) for frame in df) #in this case df would be the list of sequences 
        # Resize all DataFrames to the smallest size
        resized_sequences = [frame[0][:min_size] for frame in df]
        return resized_sequences



def remove_imbalances_old(df:pd.DataFrame,target_column:str):
    # Get counts of positive and negative values in the target column
    pos_count = df[target_column].gt(0).sum()
    neg_count = df[target_column].le(0).sum()

    # Calculate the number of rows to keep to achieve a balanced DataFrame
    rows_to_keep = min(pos_count, neg_count)

    # Separate positive and negative rows
    pos_rows = df[df[target_column] > 0].sample(rows_to_keep, replace=True)
    neg_rows = df[df[target_column] <= 0].sample(rows_to_keep, replace=True)

    # Concatenate the balanced rows
    df_balanced = pd.concat([pos_rows, neg_rows], ignore_index=True)

    return df_balanced


#transform the values of your target column to 1 if > 0 and 0 if <=0
def transform_to_binary(df:pd.DataFrame,target_column:str):
    # Check if the target column exists in the DataFrame
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # Transform the values of the target column to binary
    condition = df[target_column] > 0
    df.loc[condition,target_column] = 1
    df.loc[~condition,target_column] = 0
    return df


#this function transforms the daily return values to binary form (1 for positive and 0 for negative)
def transform_to_binary_np(targets:np.ndarray[np.float32]):
    binary_array = (targets>0).astype(int)
    return binary_array

def count_ones_and_zeros(array):
    ones_count = np.count_nonzero(array==1)
    zeros_count = np.count_nonzero(array==0)
    return zeros_count,ones_count


def calculate_rsi(symbol: str, period: int = 14):
    def relative_strength(data):
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        return rs
    
    equity_df = equity_formatter(symbol, nominal=True).drop(columns=['Volume'])
    equity_df['Close'] = pd.to_numeric(equity_df['Close'], errors='coerce')
    equity_df.dropna(subset=['Close'], inplace=True)
    
    rs = relative_strength(equity_df['Close'])
    rsi = 100 - (100 / (1 + rs))
    
    equity_df['RSI'] = rsi
    rsi_df = equity_df.drop(columns=['Close'])  # Drop the 'Close' column
    return rsi_df


#this isn't what you want if you're going to do it the way valkov does it. You're going to have to group by the target column then make 3 sequences, one for positive days, one for negative days, and one for neutral days.
def get_sequences(X_df:pd.DataFrame,Y_df:pd.DataFrame,sequence_size:int):
    sequences = []
    def is_single_column_dataframe(df):
        return df.shape[1] > 1 if len(df.shape) == 2 else False
    num_rows = X_df.shape[0]
    y_col = is_single_column_dataframe(Y_df)
    if num_rows!=Y_df.shape[0]:
        print("Get sequences error: X and Y must have same shape!")
        return -1
    count_rows = sequence_size
    indexer = 0
    while count_rows < num_rows:
        ind_data = X_df.iloc[indexer:count_rows,:]
        if y_col:
            target_data = Y_df.iloc[indexer:count_rows,:]
        else:
            target_data = Y_df.iloc[indexer:count_rows]
        sequences.append((ind_data,target_data))
        count_rows+=sequence_size
        indexer+=sequence_size
    return sequences


def sort_by_label(df:pd.DataFrame,target_column:str):
    sequences = []
    for id,groups in df.groupby(target_column):
        sequence_df = groups.drop(columns=[target_column])
        label = groups[target_column].iloc[0]
        sequences.append((sequence_df,label))
    return sequences

def append_formatter(data):
    data = str(data)
    output = data.replace('][',',')
    return output