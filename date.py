from datetime import datetime,timedelta
import json
import pandas as pd

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


#this function returns the min and max dates in a file containing json data
def get_json_date_extrema(file_path):
    with open(file_path,'r') as file:
        json_string = file.read()
    df = pd.read_json(json_string)
    start_date = min(df['date'])
    end_date = max(df['date'])
    arr = [start_date,end_date]
    return arr


def datetime_forward_fill(df):
    #this function takes in a pandas data frame and fills in all missing daily values by taking the most recent value
    if not 'Date' in df.columns:
        return print("Error: must have 'Date' column in dataframe")