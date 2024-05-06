import numpy as np
import pickle as pkl
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn   
import time
import torch.optim as optim
import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pickle as pkl
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn   
import time
import torch.optim as optim
import torch
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def load_data(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data

# Load the data
Interaction_matrices = load_data('interaction_matrices_10binned.pkl')
spectral_data = load_data('spectra_dataset_10binned.pkl')

print(Interaction_matrices.shape)
print(spectral_data.shape)

import numpy as np
import torch

# Flatten each matrix separately then store in an array
flattened_matrices = [matrix.flatten() for matrix in Interaction_matrices]

# Stack the flattened matrices on top to give shape N x 36
flattened_matrix = torch.stack(flattened_matrices)
print(flattened_matrix.shape)

print(flattened_matrix[0:10]) ## fine. 

import torch 
import numpy

threshold = 0.5

binary_flat_matrices = (flattened_matrix >= threshold).float()


# making them flat instead of including 
# abundance as that is not important for now and requires more

for matrix in binary_flat_matrices:
  for i in range (len(matrix)):
    value = matrix[i]

print(flattened_matrix[1])
print(binary_flat_matrices[1])

import pandas as pd
import torch
import numpy as np


# need to turn tensors to pandas df then append my flattened matrices to the end.
matrix_columns = [f'PA{i // 6}PB{i % 6}' for i in range(len(flattened_matrices[0]))]

bnry_int_mat_df = pd.DataFrame(binary_flat_matrices, columns = matrix_columns)
print(bnry_int_mat_df.shape)
print(bnry_int_mat_df)
spec_df = pd.DataFrame(spectral_data)
print(spec_df.shape)

# next  concat the two together 
import pandas as pd



concat_df = pd.concat([spec_df, bnry_int_mat_df], axis =1)

# pre process data some more

X_spec = concat_df.iloc[:, :2000].values # spectra data
Y_matr = concat_df.iloc[:, 2000:].values # matrices 

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# splitting into train test val split 80, 20 
X_train, X_test, y_train, y_test = train_test_split(X_spec, Y_matr, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.2, random_state=42)


print("Training Set:", X_spec.shape, y_train.shape)
print("Validation Set:", X_val.shape, y_val.shape)
print("Test Set:", X_test.shape, y_test.shape)


import torch 
from torch.utils.data import DataLoader, TensorDataset

X_train = torch.Tensor(X_train).to(device)
X_test = torch.Tensor(X_test).to(device)
X_val = torch.Tensor(X_val).to(device)
y_val= torch.Tensor(y_val).to(device)
y_train = torch.Tensor(y_train).to(device)
y_test = torch.Tensor(y_test).to(device)


batch_size = 512
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True, dropout=0.5)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, num_layers=2, batch_first=True, dropout=0.5)
        # self.lstm4 = nn.LSTM(input_size=256, hidden_size=512, num_layers=4, batch_first=True, dropout=0.0)
        self.regressor = nn.Linear(64, 36)

    def forward(self, x):
        # forward passing
        lstm_out, _ = self.lstm1(x)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_out, _ = self.lstm3(lstm_out)
        #lstm_out, _ = self.lstm4(lstm_out)

        # only last time step
        lstm_out = lstm_out[:, -1, :]

        # pass output to lstm to fully connected later to predict all 36 values
        output = self.regressor(lstm_out)
        return output


model = LSTM().to(device)

criteria = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, )



def train_model(model, data_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Started at {time.ctime()}")
        
        epoch_loss = 0
        for spectra, labels in data_loader:
            spectra, labels = spectra.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(spectra)
            loss = criterion(output, labels)
            print(out)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(data_loader)
        
        PATH = 'cp1_new_lstm_structure.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, PATH)
        
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')
        print(f"Ended at {time.ctime()}\n")

train_model(model, train_loader, criteria, optimizer, epochs=10)

# 2m22seconds for 800 dataset epoch 1 batch size 32 total loss ~6.5 lr = 0.001
# 2m22seconds for 800 dataset epoch 1 batch size 64 loss ~7.196 lr = 0.001

