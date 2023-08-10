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


d = {'Date':[2012,2013,2014,2015],'Value':[12.0,13.0,11.4,15.6]}
df = pd.DataFrame(d)


data = {'col1': [1, 2, 3,5,3,23,3,2,4,5,2],
        'col2': [4, 5, 6,3,5,21,2,2,3,4,2],
        'col3': [7, 8, 9,2,2,3,12,2,4,3,8]}

df = pd.DataFrame(data)
from sklearn.preprocessing import MinMaxScaler
X = df.to_numpy()
print(X)
# Loop through all columns except the last one
for i in range(X.shape[1]):
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaled = scaler.fit_transform(X[:,i].reshape(-1,1))
    X[:,i] = scaled.flatten()
print(X)
'''
formatted = fn.earnings_formatter('META')
print(formatted)

print(pd.isna(formatted[1]['surprise'].iloc[0]))