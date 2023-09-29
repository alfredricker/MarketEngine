from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import sys
import functions as fn

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

from multiprocessing import cpu_count
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics import Accuracy

pl.seed_everything(42) #make results reproducible
# Load and preprocess your data
#df = pd.read_csv('DATA_SENTIMENT.csv')
df = pd.read_csv('csv_data/TRAINING-9-27.csv')
df = df.iloc[:,1:] #remove the unwanted index column
df.fillna(value=0,inplace=True)
size = df.shape[1]
df = fn.transform_to_binary(df,'Close_5d')
#df = df.iloc[:68400,:]
print(df)
#print(df)

# Split the data into training and test sets (95:5 ratio)
train_df, test_df = train_test_split(df, test_size=0.05) #IMPORTANT: change test_size back to 0.05 for larger data sets.

# Remove imbalances from training data.. this is already done with sort_by_label
balanced_train_df = fn.remove_imbalances(train_df, 'Close_5d')
print(len(balanced_train_df))

# Get X_train_balanced and Y_train_balanced as DataFrames
if 'Close_Tmr' in df.columns:
    X_train = balanced_train_df.drop(columns=['Close_Tmr','Close_5d']).to_numpy()
    X_test = test_df.drop(columns=['Close_Tmr','Close_5d']).to_numpy()
    size-=1
else:
    X_train = balanced_train_df.drop(columns=['Close_5d']).to_numpy()
    X_test = test_df.drop(columns=['Close_5d']).to_numpy()

Y_train = balanced_train_df['Close_5d'].to_numpy()
Y_test = test_df['Close_5d'].to_numpy()

batch_size=32

# ... (Data preprocessing code here)
scaler_X = StandardScaler()
#scaler_X = MinMaxScaler((-1,1))
X_train = scaler_X.fit_transform(X_train)
#print(X_train[:20])
X_test = scaler_X.transform(X_test)
#Time to convert to PyTorch tensors
X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test,dtype=torch.float32)
Y_train = torch.tensor(Y_train,dtype=torch.long) #need to use long datatype for classifiers
Y_test = torch.tensor(Y_test,dtype=torch.long)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

class TimeSeriesDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i].to(device)
        y = self.Y[i].to(device)
        return x,y
    
class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, X_train, X_test, Y_train, Y_test, batch_size):
        super().__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        self.train_dataset = TimeSeriesDataset(self.X_train,self.Y_train)
        self.test_dataset = TimeSeriesDataset(self.X_test,self.Y_test)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            #num_workers=10
        )
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            #num_workers=10
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            #num_workers=10
        )
    
num_epochs = 25

data_module = TimeSeriesDataModule(X_train,X_test,Y_train,Y_test,batch_size=batch_size)

# Define your LSTM classifier

class LSTMClassifier(torch.nn.Module):
    def __init__(self, num_features=size-1, num_classes=2, hidden_size=128, num_stacked_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = torch.nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=num_stacked_layers, batch_first=True) #dropout = 0.75
        self.classifier = torch.nn.Linear(hidden_size,num_classes)

    def forward(self, x):
        #batch_size = x.size(0)
        #h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        #c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x) #(h0, c0)) #I'm referencing a point, not a tensor
        out = self.classifier(out)
        return out

#initialize the torch accuracy class
accuracy = Accuracy(task='binary').to(device)

class PricePredictor(pl.LightningModule):
    def __init__(self,num_features:int,num_classes:int):
        super().__init__()
        self.model = LSTMClassifier(num_features,num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self,x,labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output,labels)
        return loss,output
    
    def training_step(self,batch,batch_i):
        X_batch,Y_batch = batch
        loss,output = self.forward(X_batch,labels=Y_batch)
        predictions = torch.argmax(output,dim=1)
        step_accuracy = accuracy(predictions,Y_batch)
        self.log("train_loss",loss,prog_bar=True,logger=True)
        self.log("train_accuracy",step_accuracy.item(),prog_bar=True,logger=True)
        #print(f'Train Loss: {loss:.2f}')
        #print(f'Train Accuracy: {accuracy * 100:.2f}%')
        return {"loss":loss,"accuracy":accuracy}
    
    def validation_step(self,batch,batch_i):
        X_batch,Y_batch = batch
        loss,output = self.forward(X_batch,labels=Y_batch)
        predictions = torch.argmax(output,dim=1)
        step_accuracy = accuracy(predictions,Y_batch)
        self.log("val_loss",loss,prog_bar=True,logger=True)
        self.log("val_accuracy",step_accuracy.item(),prog_bar=True,logger=True)
        #print(f'Validation Loss: {loss:.2f}')
        #print(f'Validation Accuracy: {accuracy * 100:.2f}%')
        return {"loss":loss,"accuracy":accuracy}
    
    def test_step(self,batch,batch_i):
        X_batch,Y_batch = batch
        loss,output = self.forward(X_batch,labels=Y_batch)
        predictions = torch.argmax(output,dim=1)
        step_accuracy = accuracy(predictions,Y_batch)
        self.log("test_loss",loss,prog_bar=True,logger=True)
        self.log("test_accuracy",step_accuracy.item(),prog_bar=True,logger=True)
        #print(f'Test Loss: {loss:.2f}')
        #print(f'Test Accuracy: {accuracy * 100:.2f}%')
        return {"loss":loss,"accuracy":accuracy}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
    
model = PricePredictor(num_features=size-1,num_classes=2)
model.to(device)
# Set up your data, model, and training loop

checkpoint_callback = ModelCheckpoint(
    dirpath="CHECKPOINTS",
    filename="best-checkpoint-9-28.ckpt",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

logger = TensorBoardLogger("lightning_logs",name="returns")

trainer = pl.Trainer(
    logger=logger,
    max_epochs=num_epochs,
    devices=1,
    accelerator=device,
    callbacks=[checkpoint_callback]
    #default_root_dir='CHECKPOINTS/'
)

def train_model():
    trainer.fit(model,data_module)
    torch.save(model.state_dict(),'CHECKPOINTS/CLASSIFIER-9-28.ckpt')

    #get the numpy array of the predicted values
    with torch.no_grad():
        predicted = model(X_train)[1]
        predicted_test = model(X_test)[1]

    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    threshold = 0.50
    predicted_probs= torch.softmax(predicted_test, dim=1)  # Apply softmax to get class probabilities
    predicted_probs_numpy = predicted_probs.detach().numpy()

    predicted_labels = (predicted_probs[:, 1] > threshold).long() 
    # Calculate confusion matrix
    cm = confusion_matrix(Y_test.numpy(), predicted_labels)
    # Plot confusion matrix using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    #get the numpy array of the predicted values
    predicted_prob_tuples = [tuple(prob) for prob in predicted_probs]
    
    data = {
        'Actual': Y_test.numpy(),
        'Predicted': predicted_labels,
        'Predicted-Probs': predicted_prob_tuples
    }

    df_comparison = pd.DataFrame(data)
    df_comparison.to_csv('csv_tests/comparison_classifier-9-28.csv')


#train_model()
#sys.exit()


sym_list = ['A', 'AA', 'AAL', 'AAOI', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACHR', 'ADBE', 'ADM', 'AES', 'AI', 'AKAM', 'AMAT', 'AMD', 'AMZN', 'APLD', 'AR', 'AVGO', 'AXP', 'BA', 'BAC', 'BANC', 'BAX', 'BMY', 'BOWL', 'BRK-B', 'BSX', 'BX', 'BYND', 'CARR', 'CAT', 'CHPT', 'CMCSA', 'COIN', 'COP', 'CRM', 'CROX', 'CSCO', 'CSX', 'CVNA', 'DAL', 'DELL', 'DIS', 'DISH', 'DOCU', 'DVN', 'EBAY', 'ENVX', 'ET', 'ET', 'ETSY', 'EW', 'EYE', 'FDX', 'FHN', 'FITB', 'FLEX', 'FSLY', 'FSR', 'FUBO', 'GE', 'GM', 'GOOG', 'GPS', 'HAL', 'HBAN', 'HLIT', 'HOOD', 'INTC', 'IONQ', 'IP', 'JBLU', 'JNJ', 'JPM', 'KDP', 'KEY', 'KMI', 'KO', 'KVUE', 
'LCID', 'LLY', 'LSXMK', 'LUMN', 'LUV', 'LYFT', 'MARA', 'MAT', 'MCD', 'MMM', 'MRVL', 'MSFT', 'MTCH', 'MU', 'NCLH', 'NET', 'NFLX', 'NOG', 'NOV', 'NOVA', 'NVDA', 'NYCB', 'OPCH', 'OPEN', 'OSTK', 'PARA', 'PCG', 'PEP', 'PFE', 'PG', 'PINS', 'PLUG', 'PTEN', 'PWR', 'PYPL', 'QCOM', 'QS', 'RBLX', 'RCL', 'RIOT', 'RKT', 'ROKU', 'RTX', 'RUN', 'SBUX', 'SCHW', 'SHAK', 'SIRI', 'SMPL', 'SNAP', 'SOFI', 'SOUN', 'SPCE', 'SPWR', 'TER', 'TGT', 'TTD', 'TWNK', 'U', 'UBER', 'UPS', 'UPST', 
'USB', 'V', 'VICI', 'VLO', 'VZ', 'WBA', 'WBD', 'WMT', 'WU', 'XPOF', 'ZM']
device = 'cpu'

checkpoint = torch.load('CHECKPOINTS/CLASSIFIER-9-27.ckpt')
model.load_state_dict(checkpoint)
model.to(device)
#mod = PricePredictor.load_from_checkpoint('CHECKPOINTS/best-checkpoint-9-1.ckpt')

def preprocess_eval_data(pth):
    df = pd.read_csv(pth)
    df = df.iloc[:,1:] #remove the unwanted index column
    df = df.iloc[-2] #get last row of data
    #df.fillna(value=0,inplace=True)
    #size = df.shape[1]
    X = df.to_numpy()
    X_trans = scaler_X.transform(X.reshape(1,-1))
    #X_trans = X_trans.flatten()
    
    X_tens = torch.tensor(X_trans,dtype=torch.float32)
    return X_tens


def preprocess_multiple_eval_data(sym_list):
    def sym_pth(sym):
        pth = f'csv_data/equity/{sym}-2023-09-28.csv'
        return pth
    df_list = []
    
    for sym in sym_list:
        pth = sym_pth(sym)

        data = pd.read_csv(pth)
        data['Close_5d'] = data['Close'].shift(-5)
        #data = data.drop(index=data.index[-1])
        data = data.dropna()

        data = data.iloc[:,1:]
        df_list.append(data)
    
    df = pd.concat(df_list,axis=0,ignore_index=True)
    df = fn.transform_to_binary(df,'Close_Tmr')

    X = df.drop(columns=['Close_Tmr']).to_numpy()
    Y = df['Close_Tmr'].to_numpy()
    X_trans = scaler_X.transform(X)

    X_tens = torch.tensor(X_trans,dtype=torch.float32)
    Y_tens = torch.tensor(Y,dtype=torch.long)
    return (X_tens,Y_tens)

# Ensure the model is in evaluation mode
model.eval()

#trading_dict = {'Ticker':[],'Predicted':[],'PredictedProbs':[]}

def stock_pth(symbol):
    pth = f'csv_data/equity/{symbol}-2023-09-28.csv'
    return pth

trading_dict = {'Ticker':[],'Predicted':[],'PredictedProbs':[]}

'''
for sym in sym_list:
    pth = stock_pth(sym)
    data = preprocess_eval_data(pth)
    data.to(device)
    print(sym)
    # Pass the new data through the model
    with torch.no_grad():
        output = model(data)
        #print(output)
    
    # Apply softmax to get predicted probabilities
    threshold = 0.50
    predicted_probs = torch.softmax(output[1], dim=1)  # Apply softmax to get class probabilities
    prob_tuple = tuple(predicted_probs)
    predicted_labels = (predicted_probs[:, 1] > threshold).long()

    trading_dict['Ticker'].append(sym)
    trading_dict['Predicted'].append(predicted_labels)
    trading_dict['PredictedProbs'].append(prob_tuple)

df_eval = pd.DataFrame(trading_dict)
df_eval.to_csv('csv_data/EVAL-9-28.csv')
'''


data = preprocess_multiple_eval_data(sym_list)
X_dat = data[0]
Y_dat = data[1]
X_dat.to(device)

with torch.no_grad():
    output = model(X_dat)[1]
    #print(output)

threshold = 0.50
predicted_probs = torch.softmax(output, dim=1)  # Apply softmax to get class probabilities
predicted_labels = (predicted_probs[:, 1] > threshold).long() 
predicted_prob_tuples = [tuple(prob) for prob in predicted_probs]

file_data = {
        'Actual': Y_dat.numpy(),
        'Predicted': predicted_labels,
        'Predicted-Probs': predicted_prob_tuples
    }
df_eval = pd.DataFrame(file_data)
df_eval.to_csv('csv_tests/EVAL-9-27.csv')
