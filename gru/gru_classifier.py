import numpy as np
import torch
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataloader import DataLoader

class GruClassifier:
    def __init__(self, model):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if type(model) == str:
            self.model = torch.load(model, weights_only=False, map_location=self.device)
        else:
            self.model = model
        
        self.dataloader = DataLoader()
    
    def classify_level(self, level):
        _, test_data = self.dataloader.prepare_data(level, average_ref_data=False)

        results = []
        for seq in test_data:
            seq = torch.tensor(seq).unsqueeze(0).to(self.device)

            outputs = self.model(seq)
            _, predicted = torch.max(outputs.data, 1)
            results.append(predicted.item())

        return np.array(results)

