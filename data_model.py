import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 32)
        self.out = nn.Linear(32, 3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x


        