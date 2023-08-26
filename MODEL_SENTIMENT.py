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

pl.seed_everything(45) #make results reproducible

# Load and preprocess your data
#df = pd.read_csv('DATA_SENTIMENT.csv')
df = pd.read_csv('csv_data/CLASSIFIER_1.csv')
df = df.iloc[:,1:] #remove the unwanted index column
df.fillna(value=0,inplace=True)
size = df.shape[1]
df = fn.transform_to_binary(df,'Close_Tmr')
print(df)
#print(df)

# Split the data into training and test sets (95:5 ratio)
train_df, test_df = train_test_split(df, test_size=0.05)

# Remove imbalances from training data.. this is already done with sort_by_label
balanced_train_df = fn.remove_imbalances(train_df, 'Close_Tmr')

# Get X_train_balanced and Y_train_balanced as DataFrames
X_train = balanced_train_df.drop(columns=['Close_Tmr']).to_numpy()
Y_train = balanced_train_df['Close_Tmr'].to_numpy()
X_test = test_df.drop(columns=['Close_Tmr']).to_numpy()
Y_test = test_df['Close_Tmr'].to_numpy()

batch_size=32

# ... (Data preprocessing code here)
scaler_X = StandardScaler()
#scaler_X = MinMaxScaler((-1,1))
X_train = scaler_X.fit_transform(X_train)
#print(X_train[:10])
X_test = scaler_X.transform(X_test)
#scaler_Y = StandardScaler()

#scaler_Y = MinMaxScaler(feature_range=(-1,1))
#Y_train = scaler_Y.fit_transform(Y_train.reshape(-1,1))
#Y_test = scaler_Y.transform(Y_test.reshape(-1,1))

#X_train = X_train.reshape(batch_size,-1,size-1)
#X_test = X_test.reshape(batch_size,-1,size-1)
#print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)
#Time to convert to PyTorch tensors
X_train = torch.tensor(X_train,dtype=torch.float32)
X_test = torch.tensor(X_test,dtype=torch.float32)
Y_train = torch.tensor(Y_train,dtype=torch.long) #need to use long datatype for classifiers
Y_test = torch.tensor(Y_test,dtype=torch.long)

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
        return (self.train_dataset,self.test_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.setup()[0],
            batch_size=self.batch_size,
            shuffle=True,
            #num_workers=10
        )
    def val_dataloader(self):
        return DataLoader(
            self.setup()[1],
            batch_size=self.batch_size,
            shuffle=False,
            #num_workers=10
        )
    def test_dataloader(self):
        return DataLoader(
            self.setup()[1],
            batch_size=self.batch_size,
            shuffle=False,
            #num_workers=10
        )
    
num_epochs = 2

data_module = TimeSeriesDataModule(X_train,X_test,Y_train,Y_test,batch_size=batch_size)

# Define your LSTM classifier

class LSTMClassifier(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_size=256, num_stacked_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers
        self.lstm = torch.nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=num_stacked_layers, batch_first=True).to(device) #dropout = 0.75
        self.classifier = torch.nn.Linear(hidden_size,num_classes)#.to(device)

    def forward(self, x):
        #batch_size = x.size(0)
        #h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        #c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x)#self.lstm(x) #(h0, c0)) #I'm referencing a point, not a tensor
        out = self.classifier(out)
        return out

#initialize the torch accuracy class
accuracy = Accuracy(task='binary').to(device)

class PricePredictor(pl.LightningModule):
    def __init__(self,num_features:int,num_classes:int):
        super().__init__()
        self.model = LSTMClassifier(num_features,num_classes).to(device)
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self,x,labels=None):
        output = self.model(x.to(device))
        loss = 0
        if labels is not None:
            loss = self.criterion(output,labels)
        return loss,output
    
    def training_step(self,batch,batch_i):
        X_batch,Y_batch = batch[0].to(device),batch[1].to(device)
        loss,output = self.forward(X_batch,labels=Y_batch)
        predictions = torch.argmax(output,dim=1)
        step_accuracy = accuracy(predictions,Y_batch)
        self.log("train_loss",loss,prog_bar=True,logger=True)
        self.log("train_accuracy",step_accuracy.item(),prog_bar=True,logger=True)
        #print(f'Train Loss: {loss:.2f}')
        #print(f'Train Accuracy: {accuracy * 100:.2f}%')
        return {"loss":loss,"accuracy":accuracy}
    
    def validation_step(self,batch,batch_i):
        X_batch,Y_batch = batch[0].to(device),batch[1].to(device)
        loss,output = self.forward(X_batch,labels=Y_batch)
        predictions = torch.argmax(output,dim=1)
        step_accuracy = accuracy(predictions,Y_batch)
        self.log("val_loss",loss,prog_bar=True,logger=True)
        self.log("val_accuracy",step_accuracy.item(),prog_bar=True,logger=True)
        #print(f'Validation Loss: {loss:.2f}')
        #print(f'Validation Accuracy: {accuracy * 100:.2f}%')
        return {"loss":loss,"accuracy":accuracy}
    
    def test_step(self,batch,batch_i):
        X_batch,Y_batch = batch[0].to(device),batch[1].to(device)
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
    
model = PricePredictor(num_features=size-1,num_classes=2)
model.to(device)
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
    #checkpoint_callback=checkpoint_callback
)

#trainer.fit(model,data_module)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for batch in data_module.train_dataloader():
        x_batch, y_batch = batch
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        loss, output = model.forward(x_batch, labels=y_batch)

        # Backpropagation and optimization
        loss.backward()
        model.configure_optimizers()['optimizer'].step()  # Perform a single optimization step
        model.configure_optimizers()['optimizer'].zero_grad()

    # Validation step
    model.eval()  # Set the model to evaluation mode
    for batch in data_module.val_dataloader():
        x_batch, y_batch = batch
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        loss, output = model.forward(x_batch, labels=y_batch)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ... (rest of your code)

# After training, get the predictions on the test set
with torch.no_grad():
    predicted_test = model(X_test.to(device))[1].to('cpu').numpy() #the index calls the output from the tuple (loss,output)
print(predicted_test)
# Calculate confusion matrix
cm = confusion_matrix(Y_test.numpy(), predicted_test)

# Plot confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#get the numpy array of the predicted values
'''
with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()
    predicted_test = model(X_test.to(device)).to('cpu').numpy()

data = {
    'Actual': Y_test,
    'Predicted': predicted_test
}

df_comparison = pd.DataFrame(data)
df_comparison.to_csv('csv_tests/comparison_classifier.csv')
'''