import torch
from torch import nn

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()

        self.h1 = nn.Linear(378, 200) 
        self.h2 = nn.Linear(200, 50)
        self.output = nn.Linear(50, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.batchnorm1 = nn.BatchNorm1d(200)
        self.batchnorm2 = nn.BatchNorm1d(50)
        
    def forward(self, x):
        x = self.h1(x)
        x = self.relu(x)
        x = self.batchnorm1(x)
        
        x = self.h2(x)
        x = self.relu(x)
        x = self.batchnorm2(x)

        x = self.dropout(x)
        x = self.output(x)
        
        return x