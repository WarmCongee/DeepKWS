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


class ResidualBlock(torch.nn.Module):
    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        self.channels = channels
        
        self.conv1 = torch.nn.Conv2d(channels,channels,kernel_size=3,padding=1)
        self.conv2 = torch.nn.Conv2d(channels,channels,kernel_size=3,padding=1)
    
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x+y)

class KWSNet3(torch.nn.Module):
    def __init__(self):
        super(KWSNet3,self).__init__()
        self.conv1 = torch.nn.Conv2d(1,16,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16,32,kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)
        
        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)
        
        self.fc = torch.nn.Linear(512,8)
        
    def forward(self,x):
        # Flatten data from (n,1,28,28) to (n,784)
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size,-1)  # flatten
#         print(x.size(1))
        return self.fc(x)