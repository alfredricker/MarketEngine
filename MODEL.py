import torch
import datetime
from datetime import timedelta
import numpy as np
import pandas as pd
import json
import functions as fn

#HOUSING DATA
#ny = date.housing_formatter('NewYork')
#chicago = date.housing_formatter('Chicago')
#la = date.housing_formatter('LosAngeles')

#TREND DATA
with open('data_misc/trend_Apple.dat', 'r') as file:
    apple = file.read()
aapl_trend = pd.read_json(apple)

#FED DATA
def fed_init(code):
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
def housing_init(city):
    with open(f'data_misc/{city}_formatted.dat','r') as file:
        r = file.read()
    data = pd.read_json(r)
    return data
ny = housing_init('NewYork')
la = housing_init('LosAngeles')
chicago = housing_init('Chicago')
print("Initialized housing data")

#EQUITY/TARGET DATA
apple_sym = 'aapl'
aapl = fn.equity_formatter(apple_sym)
aapl_close = aapl[0]
aapl_volume = aapl[1]
aapl_earnings = fn.earnings_formatter(apple_sym)
aapl_reported_eps = aapl_earnings[0]
aapl_surprise_eps = aapl_earnings[1]
print("Initialized equity data")

#CONCATENATE ALL DATA INTO A SINGLE LARGE DATAFRAME
aapl_list = [aapl_close,aapl_volume,aapl_trend,aapl_reported_eps,aapl_surprise_eps,gdp,cpi,dff,m1,m1v,indpro,unrate,ny,la,chicago]
size = len(aapl_list)+3 #+3 because we eventually add the "Close_Tmr" column, as well as 2 counter columns
aapl_df = fn.concatenate_data(aapl_list)
#aapl_df.to_csv('aapl_test.csv',index=False)

#IMPORTANT: since I want to look at how the present day data effects the closing price of tomorrow, I have to copy the close column, and shift it up 1 position in a new row
aapl_df['Close_Tmr'] = aapl_df['Close'].shift(-1)
aapl_df = aapl_df.drop(index=aapl_df.index[-1]) #drop the last row because the above shift method will result in NaN values
aapl_df.set_index('Date',inplace=True)
#print(aapl_df)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu' #run this program on the gpu if it is available

aapl_np = aapl_df.to_numpy()

#verify that this scaling is appropriate
from sklearn.preprocessing import MinMaxScaler
#IMPORTANT: should create two different scalers to avoid data leakage
scaler_X = MinMaxScaler(feature_range=(-1,1))
scaler_Y = MinMaxScaler(feature_range=(-1,1))
#aapl_np = scaler.fit_transform(aapl_np)

#close_tmr is the value we want to predict and it is the last column of the dataframe
X = aapl_np[:,:-1]
Y = aapl_np[:,-1]
X = scaler_X.fit_transform(X)
Y = scaler_Y.fit_transform(Y.copy().reshape(-1,1))
split_index = int(len(X)*0.9) # I will use 90% of the data as training
X_train = X[:split_index]
X_test = X[split_index:]
Y_train = Y[:split_index]
Y_test = Y[split_index:]
#print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
#PyTorch LSTMs must have an extra dimension at the end
X_train = X_train.reshape(-1,size-1,1)
X_test = X_test.reshape(-1,size-1,1)
Y_train = Y_train.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)
#print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
#Time to convert to PyTorch tensors
X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test,dtype=torch.float32)
Y_train = torch.tensor(Y_train,dtype=torch.float32)
Y_test = torch.tensor(Y_test,dtype=torch.float32)

from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self,i):
        return self.X[i],self.Y[i]

train_dataset = TimeSeriesDataset(X_train,Y_train)
test_dataset = TimeSeriesDataset(X_test,Y_test)

from torch.utils.data import DataLoader
batch_size = 16
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)
#send the batches to the gpu if available
for _,batch in enumerate(train_loader):
    x_batch = batch[0].to(device)
    y_batch = batch[1].to(device)
    #print(x_batch.shape,y_batch.shape)
    break

class LSTM(torch.nn.Module):
    def __init__(self,input_size,hidden_size,num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = torch.nn.LSTM(input_size,hidden_size,num_stacked_layers,batch_first=True)
        self.fc = torch.nn.Linear(hidden_size,1)
    def forward(self,x):
        batch_size=x.size(0)
        h0 = torch.zeros(self.num_stacked_layers,batch_size,self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers,batch_size,self.hidden_size).to(device)
        out,_ = self.lstm(x,(h0,c0))
        out = self.fc(out[:,-1,:])
        return out
#first try num_stacked_layers = 2 (read into stacking layers in LSTM networks)

model = LSTM(1,4,2)
model.to(device)

#I want accuracy based loss instead of MSE
class AccuracyCrossEntropyLoss(torch.nn.Module):
    def __init__(self, accuracy_weight=1.0, cross_entropy_weight=1.0):
        super(AccuracyCrossEntropyLoss, self).__init__()
        self.accuracy_weight = accuracy_weight
        self.cross_entropy_weight = cross_entropy_weight
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, predictions, targets):
        # Calculate the accuracy
        predicted_labels = torch.argmax(predictions, dim=1)
        correct_predictions = torch.eq(predicted_labels, targets).sum().item()
        accuracy = correct_predictions / targets.size(0)

        # Calculate cross-entropy loss
        cross_entropy_loss = self.cross_entropy_loss(predictions, targets)

        # Combine the losses with weights
        loss = self.accuracy_weight * (1.0 - accuracy) + self.cross_entropy_weight * cross_entropy_loss

        return loss

#includes a penalty for getting the sign wrong    
class MSELossWithPenalty(torch.nn.Module):
    def __init__(self, penalty_factor):
        super(MSELossWithPenalty, self).__init__()
        self.penalty_factor = penalty_factor

    def forward(self, predictions, targets):
        # Calculate the mean squared error (MSE) loss
        mse_loss = torch.nn.MSELoss()(predictions, targets)

        # Apply the penalty for incorrect sign predictions
        prediction_signs = torch.sign(predictions)
        target_signs = torch.sign(targets)
        
        incorrect_signs = torch.zeros_like(prediction_signs)
        for i in range(len(prediction_signs)):
            if prediction_signs[i] != target_signs[i]:
                incorrect_signs[i] = 1.0
        sign_penalty = torch.mean(incorrect_signs)
        # Combine the MSE loss and the penalty with the given factor
        total_loss = mse_loss + self.penalty_factor * sign_penalty
        return total_loss

# Assuming you have defined your neural network as 'model'
# and you have your training data and targets as 'train_data' and 'train_targets', respectively.

#loss_function = AccuracyCrossEntropyLoss()
sign_penalty = 1.0
loss_function = MSELossWithPenalty(sign_penalty)
learning_rate = 0.00005 #this worked way better than 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 10


def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch+1}')
    running_loss = 0.0
    for batch_index,batch in enumerate(train_loader):
        x_batch,y_batch = batch[0].to(device),batch[1].to(device)
        output = model(x_batch)
        loss = loss_function(output,y_batch)
        running_loss += loss#.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_index % 100 == 99:
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss{1:,.3f}'.format(batch_index+1, avg_loss_across_batches))
            running_loss = 0.

def validate_one_epoch():
    model.train(False)
    running_loss = 0.0
    for batch_index,batch in enumerate(test_loader):
        x_batch,y_batch = batch[0].to(device),batch[1].to(device)
        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output,y_batch)        
            running_loss+=loss#.item()
    avg_loss_across_batches = running_loss / len(test_loader)          
    print('Val loss: {0:.3f}'.format(avg_loss_across_batches))
    print('--------------------------------------')

for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()

#get the numpy array of the predicted values
with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()
    predicted_test = model(X_test.to(device)).to('cpu').numpy()

X_test_orig_shape = X_test.reshape(-1,size-1)
X_train_orig_shape = X_train.reshape(-1,size-1)
X_test_original = scaler_X.inverse_transform(X_test_orig_shape)
X_train_original = scaler_X.inverse_transform(X_train_orig_shape)
Y_train_original = scaler_Y.inverse_transform(Y_train)
Y_test_original = scaler_Y.inverse_transform(Y_test).flatten()

test_predictions = scaler_Y.inverse_transform(predicted_test).flatten()

#see how many times it correctly guesses the sign
ones_test = torch.ones_like(torch.tensor(Y_test_original.copy()))
hold_perform = 0
Y_test_sign = torch.sign(torch.tensor(Y_test_original.copy()))
predictions_sign = torch.sign(torch.tensor(test_predictions.copy()))
model_perform = 0
for i in range(len(Y_test_original)):
    if predictions_sign[i] == Y_test_sign[i]:
        model_perform+=1
    if ones_test[i] == Y_test_sign[i]:
        hold_perform+=1
print(f'Model performance: {model_perform} \nHold performance: {hold_perform}')

data = {
    'Actual': Y_test_original,
    'Predicted': test_predictions
}
df_comparison = pd.DataFrame(data)
df_comparison.to_csv('AAPL_comparison.csv')