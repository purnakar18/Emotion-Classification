from torch.nn import functional as F
import torch.nn as nn
import torch

class Decoder1(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim, output_dim):
        super(Decoder1, self).__init__()
        self.fc1 = nn.Linear(input_dim1, hidden_dim)
        self.fc2 = nn.Linear(input_dim2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim*2, output_dim)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, text_embedding, image_embedding):
        text_output = self.tanh(self.fc1(text_embedding))
        image_output = self.tanh(self.fc2(image_embedding))
        combined_output = torch.cat((text_output, image_output), -1 ) # Late fusion
        output = self.sigmoid(self.fc3(combined_output))
        return output