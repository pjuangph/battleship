from typing import List
import torch.nn as nn 
import torch.nn.functional as F

# Define the neural network class
class SimpleDNN(nn.Module):
    fc = []
    def __init__(self, input_size, hidden_layers:List[int], output_size):
        super(SimpleDNN, self).__init__()
        for h in hidden_layers:
            self.fc.append(nn.Linear(input_size, h))
            input_size = h
        self.out = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = x.float()
        for layer in self.fc:
            x = F.relu(layer(x))
        x = self.out(x)
        return F.softmax(x,dim=1)
