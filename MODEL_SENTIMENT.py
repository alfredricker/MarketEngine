from typing import Any
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
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
df = pd.read_csv('DATA_SENTIMENT.csv')
df = df.iloc[:,1:] #remove the unwanted index column
df.fillna(value=0,inplace=True)
size = df.shape[1]
df = fn.transform_to_binary(df,'Close_Tmr')
#print(df)
# Split the data into training and test sets (95:5 ratio)
train_df, test_df = train_test_split(df, test_size=0.10)

# Remove imbalances from training data
balanced_train_df = fn.remove_imbalances(train_df, 'Close_Tmr')
#print(balanced_train_df)

# Get X_train_balanced and Y_train_balanced as DataFrames
X_train = balanced_train_df.drop(columns=['Close_Tmr']).to_numpy()
Y_train = balanced_train_df['Close_Tmr'].to_numpy()
X_test = test_df.drop(columns=['Close_Tmr']).to_numpy()
Y_test = test_df['Close_Tmr'].to_numpy()

batch_size=32

# ... (Data preprocessing code here)
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
#scaler_Y = StandardScaler()

scaler_Y = MinMaxScaler(feature_range=(-1,1))
Y_train = scaler_Y.fit_transform(Y_train.reshape(-1,1))
Y_test = scaler_Y.transform(Y_test.reshape(-1,1))

#X_train = X_train.reshape(batch_size,-1,size-1)
#X_test = X_test.reshape(batch_size,-1,size-1)
#print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
#Time to convert to PyTorch tensors
X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test,dtype=torch.float32)
Y_train = torch.tensor(Y_train,dtype=torch.float32)
Y_test = torch.tensor(Y_test,dtype=torch.float32)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

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
            shuffle=True
        )
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
    
num_epochs = 50

data_module = TimeSeriesDataModule(X_train,X_test,Y_train,Y_test,batch_size=batch_size)

# Define your LSTM classifier

class LSTMClassifier(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=6, num_stacked_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = torch.nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=num_stacked_layers, batch_first=True) #dropout = 0.75
        self.classifier = torch.nn.Linear(hidden_size,num_classes)

    def forward(self, x):
        #batch_size = x.size(0)
        #h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        #c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x) #(h0, c0))
        out = self.classifier(out)
        return out

#initialize the torch accuracy class
accuracy = Accuracy(task='binary')

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
        return torch.optim.Adam(self.parameters(),lr=0.0001)
    
model = PricePredictor(num_features=size-1,num_classes=2)
# Set up your data, model, and training loop

checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

logger = TensorBoardLogger("lightning_logs",name="returns")

trainer = pl.Trainer(
    logger=logger,
    max_epochs=num_epochs
)

trainer.fit(model,data_module)

'''
#get the numpy array of the predicted values
with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()
    predicted_test = model(X_test.to(device)).to('cpu').numpy()

X_test_orig_shape = X_test.reshape(-1,size-1)
X_train_orig_shape = X_train.reshape(-1,size-1)
#X_test_original = scaler_X.inverse_transform(X_test_orig_shape)
#X_train_original = scaler_X.inverse_transform(X_train_orig_shape)
Y_train_original = scaler_Y.inverse_transform(Y_train)
Y_test_original = scaler_Y.inverse_transform(Y_test).flatten()

test_predictions = scaler_Y.inverse_transform(predicted_test).flatten()

data = {
    'Actual': Y_test_original,
    'Predicted': test_predictions
}

df_comparison = pd.DataFrame(data)
df_comparison.to_csv('csv_tests/comparison_sentiment.csv')
'''