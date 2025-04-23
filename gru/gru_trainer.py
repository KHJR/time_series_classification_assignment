import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataloader import DataLoader
from utils.plotter import Plotter
from gru.gru import Gru

class GruTrainer:
    def __init__(self, level):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.level = level
        self.data_loader = DataLoader()
        self.plotter = Plotter()

        reference_data, _ = self.data_loader.prepare_data(level, average_ref_data=False)
        if level == 4:
            self.train_data = torch.concat((torch.tensor(np.array(reference_data[0])), torch.tensor(np.array(reference_data[1])))).to(self.device)
            self.train_labels = torch.tensor([0]*len(reference_data[0]) + [1]*len(reference_data[1])).to(self.device)
        else:
            self.train_data = torch.tensor(np.array(reference_data)).to(self.device)
            self.train_labels = torch.tensor([0, 1]).to(self.device)

        self.model = Gru(input_size=self.train_data.shape[-1], hidden_size=2, output_size=2).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.003)

    def train(self, num_epochs, verbose=True):
        for epoch in range(num_epochs):
            self.model.train()

            outputs = self.model(self.train_data)
            loss = self.criterion(outputs, self.train_labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if verbose:
                self.plotter.update_plot(loss.item())
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    def save_model(self, path='models/gru_model'):
        path += f'_level_{self.level}.pth'
        torch.save(self.model, path)
    