import functions as fn
import pandas as pd

'''
#dates = fn.get_json_date_extrema('data_misc/losangeles.dat')
#print(dates)
with open('data_fed/gdp_us.dat','r') as file:
    json_string = file.read()
df = pd.read_json(json_string)
print(df.max(axis=0)[0])
df_fill = fn.datetime_forward_fill(df)
with open('test.csv','w') as file:
    file.write(df_fill.to_csv())
'''

d = {'Date':[2012,2013,2014,2015],'Value':[12.0,13.0,11.4,15.6]}
df = pd.DataFrame(d)
for index,value_row in df.iterrows():
    print(index)