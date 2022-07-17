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
        self.softmax = nn.Softmax(dim=0)
    def forward(self, x):
        #x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))
        return x
