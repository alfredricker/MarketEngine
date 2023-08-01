import date
import pandas as pd

#dates = date.get_json_date_extrema('data_misc/losangeles.dat')
#print(dates)
with open('data_fed/gdp_us.dat','r') as file:
    json_string = file.read()
df = pd.read_json(json_string)
print(df.columns)
print('Date' in df.columns)