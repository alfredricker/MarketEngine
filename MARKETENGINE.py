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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    
#num_epochs = 25

#data_module = TimeSeriesDataModule(X_train,X_test,Y_train,Y_test,batch_size=batch_size)

# Define your LSTM classifier

class LSTMClassifier(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=128, num_stacked_layers=2):
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
    




model = PricePredictor.load_from_checkpoint("/lightning_logs/returns/version_54/checkpoints/epoch=24-step=39125.ckpt")

