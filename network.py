import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class KWSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(40*41, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 8)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    
class KWSNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dnn_1 = nn.Linear((30 + 1 + 10) * 40, 512)
        self.bn_1 = nn.BatchNorm1d(512)
        self.dropout_1 = nn.Dropout(0.2)
        self.dnn_2 = nn.Linear(512, 256)
        self.bn_2 = nn.BatchNorm1d(256)
        self.dropout_2 = nn.Dropout(0.2)
        self.dnn_3 = nn.Linear(256, 256)
        self.bn_3 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, 8)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        #x = x.view(x.shape[0],-1)
        x = self.flatten(x)
        x = self.dnn_1(x)
        x = self.bn_1(x)
        x = self.relu(x)
        x = self.dropout_1(x)
        x = self.dnn_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.dnn_3(x)
        x = self.bn_3(x)
        x = self.relu(x)
        return self.out(x)