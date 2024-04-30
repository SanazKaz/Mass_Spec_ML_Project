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
Interaction_matrices = load_data('interaction_matrices.pkl')
spectral_data = load_data('spectra_dataset.pkl')

print(Interaction_matrices.shape)
print(spectral_data.shape)


# reshaping for LSTM - added one dimension at the end
spectral_data_tensor = torch.tensor(spectral_data.unsqueeze(-1), dtype=torch.float32).to(device)
matrices_tensor = torch.tensor(Interaction_matrices.view(-1, 36), dtype=torch.float32).to(device)

new_tensor = spectral_data_tensor[:4000]
new_matrices = matrices_tensor[:4000]
print(new_tensor.shape)
print(new_matrices.shape)

print(matrices_tensor.shape)
print(spectral_data_tensor.shape)

print(spectral_data.dtype)
print(Interaction_matrices.dtype)

# Splitting the data
spec_train, spec_test, matrix_train, matrix_test = train_test_split(new_tensor, new_matrices, test_size=0.2, random_state=42)
spec_test, spec_val, matrix_test, matrix_val = train_test_split(spec_test, matrix_test, test_size=0.5, random_state=42)


# reshaped for LSTM
print(spec_train.shape)
print(spec_test.shape)
print(spec_val.shape)

print(matrix_train.shape)
print(matrix_test.shape)
print(matrix_val.shape)



batch_size = 64
# pytorch dataset and loaders
train_dataset = TensorDataset(torch.Tensor(spec_train), torch.Tensor(matrix_train))
val_dataset = TensorDataset(torch.Tensor(spec_val), torch.Tensor(matrix_val))
test_dataset = TensorDataset(torch.Tensor(spec_test), torch.Tensor(matrix_test))

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

