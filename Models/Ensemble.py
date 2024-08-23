from torch.nn import functional as F
import torch.nn as nn
class Decoder3(nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim):
        super(Decoder3, self).__init__()
        self.fc_t = nn.Linear(input_dim1, 256)

        self.fc_i = nn.Linear(input_dim2, 256)


        self.fc2 = nn.Linear(256, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, output_dim)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, text_embedding, image_embedding):
        text_output = self.tanh(self.fc_t(text_embedding))
        image_output = self.tanh(self.fc_i(image_embedding))
        combined_output = (3*text_output + image_output)/4
        x = self.fc2(combined_output)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

