import mat73
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.metrics import mean_squared_error
import math
from scipy.stats import pearsonr
import time

start_time = time.time()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
batch_size = 11
hidden_size = 60
dropout_ratio = 0.3
num_layers = 1
mylr = 0.005
epochs = 20


def train_test_split_func():
    # load raw data
    filename = '/storage1/jian/Active/dong/Multi_year_CCN_LSTM_new_10days_newCCNdata_v2.mat'
    raw_data = mat73.loadmat(filename)
    time_traj = raw_data['time_valid']

    data_input_norm = np.load('/storage1/jian/Active/dong/data_input_norm.npy')
    CCN_valid = np.load('/storage1/jian/Active/dong/CCN_valid.npy')

    test_flag = (time_traj[:, 0] == 2021) & (time_traj[:, 1] % 2 == 0)
    train_flag = (time_traj[:, 0] != 2021) | (time_traj[:, 1] % 2 != 0)

    x_train = data_input_norm[train_flag, :, :]
    y_train = CCN_valid[train_flag]
    x_test = data_input_norm[test_flag, :, :]
    y_test = CCN_valid[test_flag]

    train_pairs = list(zip(x_train, y_train))
    test_pairs = list(zip(x_test, y_test))

    return train_pairs, test_pairs


class DataPairsDataset(Dataset):
    def __init__(self, data_pairs):
        self.data_pairs = data_pairs
        self.sample_len = len(data_pairs)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, index):
        index = min(max(index, 0), self.sample_len - 1)

        x = self.data_pairs[index][0]
        y = self.data_pairs[index][1]

        tensor_x = torch.tensor(x, dtype=torch.float, device=device)
        tensor_y = torch.tensor(y, dtype=torch.float, device=device)

        return tensor_x, tensor_y


class LSTM_ATT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, batch_size):
        super(LSTM_ATT, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, num_layers=num_layers,
                            batch_first=True)
        self.linear1 = nn.Linear(self.hidden_size, 1)
        self.linear2 = nn.Linear(20, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_ratio)

    def forward(self, input, hidden, c):
        # input(batch_size, seq_length, input_size)
        rr, (hn, c) = self.lstm(input, (hidden, c))
        # rr(batch_size, seq_len, hidden_size)
        # hn(num_layers, batch_size, hidden_size)
        # cn(num_layers, batch_size, hidden_size)
        output = rr[:, -1, :]
        output = self.dropout(output)
        output = self.linear1(output)
        # output = self.linear2(self.relu(output))

        output = output.to(device)
        hn = hn.to(device)
        c = c.to(device)

        return output, hn, c

    def inithiddenAndC(self):
        c0 = torch.zeros(num_layers, self.batch_size, self.hidden_size, device=device)
        h0 = torch.zeros(num_layers, self.batch_size, self.hidden_size, device=device)
        return h0, c0


def LSTM_train():
    train_pairs, test_pairs = train_test_split_func()

    train_dataset = DataPairsDataset(train_pairs)
    test_dataset = DataPairsDataset(test_pairs)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    LSTM_model = LSTM_ATT(25, hidden_size, 1, num_layers, batch_size).to(device)
    myadam = torch.optim.Adam(LSTM_model.parameters(), lr=mylr)
    mse_loss = nn.MSELoss()

    train_rmse_loss_final = 10
    test_rmse_loss_final = 10
    train_r2score_final = 0
    test_r2score_final = 0
    test_y_pre_final = []

    for epoch_idx in range(1, epochs + 1):
        train_y_true = []
        train_y_pre = []
        test_y_true = []
        test_y_pre = []

        LSTM_model.train()
        for train_item, (train_x, train_y) in enumerate(tqdm(train_dataloader), start=1):
            train_h0, train_c0 = LSTM_model.inithiddenAndC()
            train_output, train_hidden, train_c = LSTM_model(train_x, train_h0, train_c0)
            # print(train_output.size())
            myadam.zero_grad()
            train_loss = mse_loss(train_output.squeeze(), train_y)
            train_loss.backward()
            myadam.step()

            train_y_true.extend(train_y.squeeze().tolist())
            train_y_pre.extend(train_output.squeeze().tolist())

        LSTM_model.eval()
        with torch.no_grad():
            for test_item, (test_x, test_y) in enumerate(test_dataloader, start=1):
                test_h0, test_c0 = LSTM_model.inithiddenAndC()
                test_predict, test_hidden, test_c = LSTM_model(test_x, test_h0, test_c0)

                test_y_true.extend(test_y.squeeze().tolist())
                test_y_pre.extend(test_predict.squeeze().tolist())

        train_rmse_loss = np.sqrt(mean_squared_error(train_y_true, train_y_pre))
        train_r2score = math.pow(pearsonr(train_y_true, train_y_pre)[0], 2)
        test_rmse_loss = np.sqrt(mean_squared_error(test_y_true, test_y_pre))
        test_r2score = math.pow(pearsonr(test_y_true, test_y_pre)[0], 2)

        print(f'The result of epoch{epoch_idx}:')
        print("Test RMSELoss:", test_rmse_loss)
        print('Test R2', test_r2score)
        print("Train RMSELoss:", train_rmse_loss)
        print("Train R2:", train_r2score)
        print("*" * 50)
        if test_rmse_loss < test_rmse_loss_final:
            test_rmse_loss_final = test_rmse_loss
            test_y_pre_final = test_y_pre

    np.save('output_save/LSTM_test_y_pre.npy', np.array(test_y_pre_final))
    end_time = time.time()
    time_consuming = end_time - start_time
    print(f'Time consuming:{time_consuming:.2f}s')


if __name__ == '__main__':
    LSTM_train()
