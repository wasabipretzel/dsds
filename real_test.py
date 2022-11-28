import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import json
import os
import logging

PATH = os.getcwd()


with open(PATH + '/data/preprocessed/3_month_retail.json', 'r') as f:
    data = json.load(f)

with open(PATH + '/data/preprocessed/cluster_store.json','r') as f:
    cluster = json.load(f)
with open(PATH + '/data/preprocessed/cluster_key_store_num.json', 'r') as f:
    cluster_key_store = json.load(f)



def cluster_dataset(cluster_num):
    data_tensor = torch.tensor([])
    for store_id in cluster[cluster_num]:
        sub_tensor = torch.tensor(list(data[str(store_id)].values()))
        data_tensor = torch.cat((data_tensor,sub_tensor),0)
    return data_tensor[:,:-1].t(), data_tensor[:,-1].t()


class DSModel0(nn.Module):
    def __init__(self):
        super(DSModel0, self).__init__()
        
        #self.ln = nn.LayerNorm(7848)
        self.conv_1 = nn.Conv2d(1, 1, kernel_size=(4,4))

        self.fc1 = nn.Linear(324, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8,1)
    
    def forward(self, x):
        #x = x.reshape(1, 1, 7848)
        #x = self.ln(x)
        x = x.reshape(1, 1, 18, 436)
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, (4, 4))
        x = x.reshape(1,324)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return x

class DSModel1(nn.Module):
    def __init__(self):
        super(DSModel1, self).__init__()
        
        #self.ln = nn.LayerNorm(1098)
        self.conv_1 = nn.Conv2d(1, 1, kernel_size=(4,4))

        self.fc1 = nn.Linear(42, 8)
        self.fc2 = nn.Linear(8,1)
    
    def forward(self, x):
        #x = x.reshape(1, 1, 1098)
        #x = self.ln(x)
        x = x.reshape(1, 1, 18, 61)
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, (4, 4))
        x = x.reshape(1, 42)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        return x

class DSModel2(nn.Module):
    def __init__(self):
        super(DSModel2, self).__init__()
        
        #self.ln = nn.LayerNorm(2610)
        self.conv_1 = nn.Conv2d(1, 1, kernel_size=(4,4))

        self.fc1 = nn.Linear(105, 32)
        self.fc2 = nn.Linear(32,8)
        self.fc3 = nn.Linear(8,1)
    
    def forward(self, x):
        #x = x.reshape(1,1, 2610)
        #x = self.ln(x)
        x = x.reshape(1, 1, 18, 145)
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x, (4, 4))
        x = x.reshape(1,105)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return x


if __name__ == '__main__':
    logging.basicConfig(filename='logging_list.txt', level=logging.DEBUG, format=' % (asctime)s - %(levelname)s - %(message)s')

    store_list = [*range(1,643)]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result_tensor = torch.tensor([]).to(device)

    for store_id in store_list:
        #상점이 어느 cluster에 해당하는지
        cluster_num = cluster_key_store[str(store_id)]
        # X = cluster_dataset(cluster_num)
        # X_train = X[:27]
        # X_test = X[27:]
        X_train, X_test = cluster_dataset(cluster_num)
        # X_train, X_test size : (28, ~)

        # scaling X
        scaler = StandardScaler()
        X_train = torch.tensor(scaler.fit_transform(X_train))
        X_test = torch.tensor(scaler.transform(X_test.view(1,-1))).flatten()

        X_train = X_train.to(device, dtype=torch.float)
        X_test = X_test.to(device, dtype=torch.float)
        #product
        loss_ = {}
        for item in data[str(store_id)].keys():
            y = torch.tensor(data[str(store_id)][item][1:]).float()
            y_train = y 

            scaler2 = StandardScaler()
            y_train = torch.tensor(scaler2.fit_transform(pd.DataFrame(y_train)).flatten())

            y_train = y_train.to(device, dtype=torch.float)
            dataset_train = TensorDataset(X_train, y_train)
            train_dataloader = DataLoader(dataset_train, batch_size=1, shuffle=False)

            if cluster_num == '0':
                model = DSModel0()
            elif cluster_num == '1':
                model = DSModel1()
            elif cluster_num == '2':
                model = DSModel2()


            model = model.to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay=1e-7)

            # item_loss = 0.0
            for epoch in range(5):
                # running_loss = 0.0
                for i, data_input in enumerate(train_dataloader,0):
                    inputs, value = data_input
                    optimizer.zero_grad()
                    outputs = model(inputs).flatten()
                    loss = criterion(outputs, value)
                    loss.backward()
                    optimizer.step()

                    # running_loss += loss.item()

            #item inference
            with torch.no_grad():
                model.eval()
                y_pred = model(X_test)

                # output 역으로 바꿔줘야함
                y_pred = torch.tensor(scaler2.inverse_transform(y_pred.detach().cpu().numpy()))
                y_pred = y_pred.to(device, dtype=torch.float)
                #[rint("store : ", store_id, " item : ", item ," Pred : ", y_pred)
            result_tensor = torch.cat((result_tensor, y_pred),0)
    

    result_tensor_df = pd.DataFrame(result_tensor.detach().cpu().numpy())
    #result_tensor_df = result_tensor_df.round(0).astype(int)

    result_tensor_df.to_csv('draft_epoch5.csv')

    logging.debug('End of Program')
        
