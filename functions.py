from datetime import datetime,timedelta
import json
import pandas as pd

#changes the values column of a pandas data frame to be percentage instead of nominal
#this function only works for updating one column
def df_percent_change(df,target):
    if not 'Date' in df.columns:
        print("Must have date and value column in df for percent_change()")
    df_sort = df.sort_values(by='Date')

    df_sort[target] = df_sort[target].astype(float)
    df_sort[target] = df_sort[target].pct_change()
    df_sort_copy = df_sort.copy()
    df_sort_copy.loc[df_sort_copy.index[0], target] = df_sort_copy.loc[df_sort_copy.index[1], target]
    #df_sort[target].iloc[0] = df_sort[target].iloc[1] #as not to have NaN in the first column
    #df_sort = df_sort.drop(labels=df_sort.index[0],axis=0)
    return df_sort_copy


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


#works for more than two columns of data
def datetime_forward_fill(df):
    #this function takes in a pandas data frame and fills in all missing daily values by taking the most recent value
    if not 'Date' in df.columns:
        return print("Error: must have 'Date' column in dataframe")
    df_sorted = df.sort_values(by='Date')
    df_sorted['Date'] = pd.to_datetime(df_sorted['Date'])
    currentdate = df_sorted['Date'].iloc[0]
    loc_index = 1
    df_new = df_sorted
    while currentdate<df_sorted.max(axis=0)[0]:
        #check to see if next date has data     
        if currentdate + timedelta(days=1) == df_sorted['Date'].iloc[loc_index]:
            currentdate = df_sorted['Date'].iloc[loc_index]
            loc_index+=1
        else:
            #loop through all target columns
            inc_date = currentdate + timedelta(days=1)
            new_row = {'Date':inc_date}
            for column in df.columns[1:]:
                new_row[column] = df_sorted[column].iloc[loc_index-1]
            df_new = pd.concat([df_new,pd.DataFrame([new_row])],ignore_index=True)
            currentdate = inc_date
    df_new = df_new.sort_values(by='Date')
    return df_new


#includes a column that tells you how many weeks since the updated data
def forward_fill_with_counter(df):
    #this function takes in a pandas data frame and fills in all missing daily values by taking the most recent value
    if not 'Date' in df.columns:
        return print("Error: must have 'Date' column in dataframe")
    df_sorted = df.sort_values(by='Date')
    df_sorted['Date'] = pd.to_datetime(df_sorted['Date'])
    currentdate = df_sorted['Date'].iloc[0]
    loc_index = 1
    df_new = df_sorted
    day_counter = 0
    week_counter=0
    df_sorted['Counter']=0 #create a new column filled with zeros
    while currentdate<df_sorted.max(axis=0)[0]:
        #check to see if next date has data     
        if currentdate + timedelta(days=1) == df_sorted['Date'].iloc[loc_index]:
            currentdate = df_sorted['Date'].iloc[loc_index]
            loc_index+=1
            day_counter = 0
            week_counter = 0
        else:
            #loop through all target columns
            inc_date = currentdate + timedelta(days=1)
            day_counter+=1
            if day_counter%7==0:
                week_counter+=1
            new_row = {'Date':inc_date,'Counter':week_counter}
            for column in df.columns[1:]:
                new_row[column] = df_sorted[column].iloc[loc_index-1]
            df_new = pd.concat([df_new,pd.DataFrame([new_row])],ignore_index=True)
            currentdate = inc_date
    df_new = df_new.sort_values(by='Date')
    return df_new


#works for more than two columns of data
def percent_forward_fill(df):
    #this function takes in a pandas data frame and fills in all missing daily values by taking the most recent value
    if not 'Date' in df.columns:
        return print("Error: must have 'Date' column in dataframe")
    df_sorted = df.sort_values(by='Date')
    df_sorted['Date'] = pd.to_datetime(df_sorted['Date'])
    currentdate = df_sorted['Date'].iloc[0]
    loc_index = 1
    #change the data from nominal to percentage
    for column in df.columns[1:]:
        df_sorted = df_percent_change(df,column)
    df_sorted['Date'] = pd.to_datetime(df_sorted['Date'])
    df_new = df_sorted
    while currentdate<df_sorted['Date'].max(axis=0):
        #check to see if next date has data     
        if currentdate + timedelta(days=1) == df_sorted['Date'].iloc[loc_index]:
            currentdate = df_sorted['Date'].iloc[loc_index]
            loc_index+=1
        else:
            #loop through all target columns
            inc_date = currentdate + timedelta(days=1)
            new_row = {'Date':inc_date}
            for column in df.columns[1:]:
                new_row[column] = df_sorted[column].iloc[loc_index-1]

            df_new = pd.concat([df_new,pd.DataFrame([new_row])],ignore_index=True)
            currentdate = inc_date
    df_new = df_new.sort_values(by='Date')
    return df_new


#similar to forward fill but tailored to the zillow housing data
#this also rewrites the value column as a percent change
def housing_formatter(city):
    with open(f'data_misc/{city}.dat','r') as f:
        f_read = f.read()
    df = pd.read_json(f_read)
    df_sorted = df.sort_values(by='date')
    df = pd.DataFrame({'Date':df_sorted['date'],'Value':df_sorted['value']})
    df = df_percent_change(df,'Value')
    currentdate = df['Date'].iloc[0]
    loc_index = 1
    c2 = str(df.columns[1]) #get the name of the second column
    #forward fill loop
    while currentdate<df.max(axis=0)[0]:    
        if currentdate + timedelta(days=1) == df['Date'].iloc[loc_index]:
            currentdate = df['Date'].iloc[loc_index]
            loc_index+=1
        else:
            inc_date = currentdate + timedelta(days=1)
            new_row = {'Date':inc_date,f'{c2}':df[f'{c2}'].iloc[loc_index-1]}
            df = pd.concat([df,pd.DataFrame([new_row])],ignore_index=True)
            currentdate = inc_date
    df_sorted = df.sort_values(by='Date')
    return df_sorted


#this function makes all your dataframe timelines line up. input a list of pandas data frames
def get_df_date_extrema(df_list):
    min_dates = []
    max_dates = []
    for df in df_list:
        df['Date'] = pd.to_datetime(df['Date'])
        min = df['Date'].min(axis=0,skipna=True)#[0]
        min_dates.append(min)
        max = df['Date'].max(axis=0,skipna=True)
        #print(f'min_date:{min}, max_date:{max}')
        max_dates.append(max)
    #now get the largest min dates and the smallest max dates
    min_date = datetime(1900,1,1)
    max_date = datetime(2024,1,1)
    for min in min_dates:
        if min>min_date:
            min_date = min
    for max in max_dates: 
        if max<max_date:
            max_date = max
    #print(f'[{min_date},{max_date}]')
    return [min_date,max_date]


#properly reads and formats the pd dataframe, forward fills the data, and splits data into a [price,volume] data list
def equity_formatter(symbol):
    with open(f'data_equity/{symbol}.dat','r') as file:
        d = file.read()
    j = json.loads(d)
    dat = j['Time Series (Daily)'] #get rid of the MetaData classifier in the json file
    df = pd.DataFrame(dat).T #create a dataframe and transpose it
    df.reset_index(inplace=True) #this line makes it so that the date becomes an accessible column rather than the index
    column_names = ['Date','Open','High','Low','Close','Volume']
    df.columns = column_names #rename the columns
    #split volume df and price data
    volume_df = df[['Date','Volume']]
    volume_df = datetime_forward_fill(volume_df)
    price_df = df[['Date','Close']] #I just want close for the first trial of this. I will come back and do stuff with high and low data as well
    price_df = percent_forward_fill(price_df)
    print(f'Successfully formatted {symbol} data')
    return [price_df,volume_df]


#for now I'm just going to return dfs of the report date, reportedEPS(as percent change) and the report date, surprise value (with counter forward fills)
def earnings_formatter(symbol):
    with open(f'data_equity/{symbol}_earnings.dat','r') as file:
        d = file.read()
    j = json.loads(d)
    dat = j['quarterlyEarnings']
    df = pd.DataFrame(dat)
    #change reportedDate column to Date
    df.rename(columns={'reportedDate':'Date'},inplace=True)
    reported_eps = df.iloc[1:,[1,2]]#doing 1: for rows because most recent has missing values
    surprise = df.iloc[1:,[1,4]]
    reported_eps_p = df_percent_change(reported_eps,'reportedEPS')
    reported_eps_fill = forward_fill_with_counter(reported_eps_p)
    surprise_fill = forward_fill_with_counter(surprise)
    return [reported_eps_fill,surprise_fill]


def fed_formatter(code,nominal):
    with open(f'data_fed/{code}.dat','r') as file:
        d = file.read()
    df = pd.read_json(d)
    if nominal==True:
        df_fill = datetime_forward_fill(df)
    else:
        df_fill = percent_forward_fill(df)
    df_fill['Date'] = pd.to_datetime(df_fill['Date'])
    #now remove all rows with dates before 1980 to make the code faster
    df = df_fill.loc[df_fill['Date'] >= pd.to_datetime('1980-01-01')]
    print(f'Successfully formatted {code} data')
    return df_fill


#returns the (almost) proper df to start the neural network process
def concatenate_data(df_list):
    min_max = get_df_date_extrema(df_list)
    min_value = pd.to_datetime(min_max[0])
    max_value = pd.to_datetime(min_max[1])
    df_filtered = []
    for df in df_list:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date')
        df = df[df['Date']>=min_value]
        df = df[df['Date']<=max_value]
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