import torch
import torch.nn as nn
import torch.nn.functional as F

class Gru(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # Initial hidden state
        out, _ = self.gru(x, h0)  # GRU layer
        out = self.fc(out[:, -1, :])  # Fully connected layer on the last time step
        return out