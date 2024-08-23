from torch.nn import functional as F
import torch.nn as nn

class Decoder5(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder5, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        return x