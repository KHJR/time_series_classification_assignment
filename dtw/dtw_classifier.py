import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.dataloader import DataLoader
from dtw.dtw import Dtw

class DtwClassifier:
    def __init__(self):
        self.data_loader = DataLoader()
        self.dtw = Dtw()
        pass

    def classify_level(self, level):
        reference_data, test_data = self.data_loader.prepare_data(level, average_ref_data=(level == 4))

        results = []
        for test_seq in test_data:
            distances = []
            for ref_seq in reference_data:
                distance = self.dtw.compute_dtw_distance(ref_seq, test_seq)
                distances.append(distance)
            results.append(np.argmin(distances))
            print(distances)

        return np.array(results)