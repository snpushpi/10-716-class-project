import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import os
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tst import Transformer
from tqdm import tqdm
from scipy.stats import pearsonr
device = "cuda"

def create_sequences(data, input_length, output_length):
    X, y = [], []
    for i in range(len(data) - input_length - output_length):
        X.append(data[i:(i + input_length)])
        # y.append(data[(i + input_length):(i + input_length + output_length), 2])  # Extracting only the 'Close' values
        y.append(data[(i + input_length):(i + input_length + output_length), 2:3])  # 2 is the index of 'Close' in input_features
        # print(y)
    X = np.array(X)
    y = np.array(y)
    return X, y

def training_dataset_preprocess(training_dataset_name,input_length,output_length,feature_columns):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names_1 = training_dataset_name
    names = names_1
    num_stocks = 1
    
    csv_data = pd.read_csv(os.path.join("/home/spushpit/FNSPID/stock_news_author_int_final", names_1))
    symbol_name = names_1.split('.')[0]
    print(csv_data.columns)
    csv_data = csv_data.dropna(subset=['Potency'])
    data = csv_data[feature_columns].values
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Split training data into training and validation sets
    split_ratio = 0.85
    split = int(split_ratio * len(scaled_data))
    data_train = scaled_data[:split]
    data_test = scaled_data[split:]
    
    # Splitting the dataset into training and testing sets (80-20 split)
    X_train, y_train = create_sequences(data_train, input_length, output_length)
    X_test, y_test = create_sequences(data_test, input_length, output_length)
    
    # Displaying the shapes of the datasets to ensure correctness
    print('X_train: ',X_train.shape, 'X_test', X_test.shape, 'y_train', y_train.shape, 'y_test',y_test.shape)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)  # Transposing to match model's input shape
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    batch_size = 64  
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return dataloader_train, dataloader_test, scaler

def train(input_length, output_length, dataloader_train, scaler):
    d_input = input_length 
    d_output = output_length 
    d_model = 32 # Lattent dim
    q = 8 # Query size
    v = 8 # Value size
    h = 8 # Number of heads
    N = 4 # Number of encoder and decoder to stack
    attention_size = 30 # Attention window size
    dropout = 0.1 # Dropout rate
    pe = 'regular' # Positional encoding
    chunk_mode = None
    epochs = 50
    
    # Creating the model
    model = Transformer(d_input, d_model, d_output, q, v, h, N, attention_size=attention_size, dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
    # model = TimeSeriesTransformer(num_features, num_outputs, dim_val, n_heads, n_decoder_layers, dropout_rate).to(device)
    # Loss function and optimizer
    loss_function = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()
    # Prepare loss history
    hist_loss = np.zeros(epochs)
    for idx_epoch in range(epochs):
        running_loss = 0
        with tqdm(total=len(dataloader_train.dataset), desc=f"[Epoch {idx_epoch+1:3d}/{epochs}]") as pbar:
            for idx_batch, (x, y) in enumerate(dataloader_train):
                y = y.squeeze(-1)
                optimizer.zero_grad()
                netout = model(x.to(device))
                loss = loss_function(y.to(device), netout)
                loss.backward()
                optimizer.step()   
                running_loss += loss.item()  
            train_loss = running_loss/len(dataloader_train) 
            print(train_loss)
            hist_loss[idx_epoch] = train_loss
            
    print("Training complete.")            
    plt.plot(hist_loss, 'o-', label='train')
    plt.legend()
    return model

def eval_model(model, dataloader_test, scaler, output_length):
      predictions = []
      actuals = []
      model.eval()
      with torch.no_grad(): 
        # for x, y in enumerate(dataloader_test):
        for x, y in dataloader_test:
          y = y.squeeze(-1)
          modelout_pre  = model(x.to(device))
          modelout = modelout_pre
          predictions.append(modelout.cpu().numpy())
          actuals.append(y.cpu().numpy())
      y_test_origin, y_pred_origin = np.concatenate(actuals), np.concatenate(predictions)
      y_pred_origin = y_pred_origin.reshape(-1)
      y_test_origin = y_test_origin.reshape(-1)
      mse = mean_squared_error(y_test_origin, y_pred_origin) 
      mae = mean_absolute_error(y_test_origin, y_pred_origin)
      corr, _ = pearsonr(y_test_origin, y_pred_origin)
      r2 = r2_score(y_test_origin, y_pred_origin)
      return mse, mae, corr

output_lengths = [1,2,3,4,5,6]
seq_length = 50
mse_list = {}
mae_list = {}
r2_list ={}
training_dataset_names = [f for f in os.listdir("/home/spushpit/FNSPID/stock_news_author_int_final") if f.endswith('.csv')]
feature_columns = ['Volume', 'Open', 'Close', 'Scaled_sentiment']
input_length = len(feature_columns)
for output_length in output_lengths:
    mse_list[output_length] = []
    mae_list[output_length] = []
    r2_list[output_length] = []
    for training_dataset_name in training_dataset_names:
        dataloader_train, dataloader_test, scaler = training_dataset_preprocess(training_dataset_name,seq_length,output_length,feature_columns)
        model = train(input_length, output_length, dataloader_train, scaler)
        mse, mae, r2 = eval_model(model, dataloader_test, scaler, output_length)
        mse_list[output_length].append(mse)
        mae_list[output_length].append(mae)
        r2_list[output_length].append(r2)
print(mae_list,flush=True)
print(mse_list,flush=True)
print(r2_list,flush=True)


training_dataset_names = [f for f in os.listdir("/home/spushpit/FNSPID/stock_news_author_int_final") if f.endswith('.csv')]
feature_columns = ['Volume', 'Open', 'Close']
mse_list = {}
mae_list = {}
r2_list = {}
input_length = len(feature_columns)
for output_length in output_lengths:
    mse_list[output_length] = []
    mae_list[output_length] = []
    r2_list[output_length] = []
    for training_dataset_name in training_dataset_names:
        dataloader_train, dataloader_test, scaler = training_dataset_preprocess(training_dataset_name,seq_length,output_length,feature_columns)
        model = train(input_length, output_length, dataloader_train, scaler)
        mse, mae, r2 = eval_model(model, dataloader_test, scaler, output_length)
        mse_list[output_length].append(mse)
        mae_list[output_length].append(mae)
        r2_list[output_length].append(r2)
print(mae_list,flush=True)
print(mse_list,flush=True)
print(r2_list,flush=True)

feature_columns = ['Volume', 'Open', 'Close','Potency']
mse_list = {}
mae_list = {}
r2_list = {}
corr_list = {}
input_length = len(feature_columns)
for output_length in output_lengths:
    mse_list[output_length] = []
    mae_list[output_length] = []
    r2_list[output_length] = []
    corr_list[output_length] = []
    for training_dataset_name in training_dataset_names:
        dataloader_train, dataloader_test, scaler = training_dataset_preprocess(training_dataset_name,seq_length,output_length,feature_columns)
        model = train(input_length, output_length, dataloader_train, scaler)
        mse, mae, r2 = eval_model(model, dataloader_test, scaler, output_length)
        mse_list[output_length].append(mse)
        mae_list[output_length].append(mae)
        r2_list[output_length].append(r2)
print(mae_list,flush=True)
print(mse_list,flush=True)
print(r2_list,flush=True)



feature_columns = ['Volume', 'Open', 'Close','Potency','Scaled_sentiment']
mse_list = {}
mae_list = {}
r2_list = {}
corr_list = {}
input_length = len(feature_columns)
for output_length in output_lengths:
    mse_list[output_length] = []
    mae_list[output_length] = []
    r2_list[output_length] = []
    for training_dataset_name in training_dataset_names:
        dataloader_train, dataloader_test, scaler = training_dataset_preprocess(training_dataset_name,seq_length,output_length,feature_columns)
        model = train(input_length, output_length, dataloader_train, scaler)
        mse, mae, r2 = eval_model(model, dataloader_test, scaler, output_length)
        mse_list[output_length].append(mse)
        mae_list[output_length].append(mae)
        r2_list[output_length].append(r2)
print(mae_list,flush=True)
print(mse_list,flush=True)
print(r2_list,flush=True)
