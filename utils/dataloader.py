import numpy as np
import os

class DataLoader:
    def __init__(self):
        pass

    def load(self, path):
        data = []
        with open(path, 'r') as file:
            for line in file:
                data.append(line.strip().split('\t'))
        
        return np.array(data, dtype=np.float32)

    def load_all(self, path):
        data = []
        for file_name in os.listdir(path):
            if file_name.endswith('.dat'):
                file_path = os.path.join(path, file_name)
                data.append(self.load(file_path))
        
        return data

    def prepare_data(self, level, average_ref_data=False):
        if 1 <= level <= 3:
            # Load reference data
            reference_path = f'dataset/level{level}/reference'
            reference_data = self.load_all(reference_path)

            # Load test data
            test_path = f'dataset/level{level}/test'
            test_data = self.load_all(test_path)
        elif level == 4: # Level 4 has different file structure
            # Load reference data
            reference_path = 'dataset/level4/reference'
            reference_data_1 = np.array(self.load_all(reference_path + '/1'))
            reference_data_2 = np.array(self.load_all(reference_path + '/2'))

            if average_ref_data:
                reference_data_1 = np.mean(reference_data_1, axis=0)
                reference_data_2 = np.mean(reference_data_2, axis=0)

            reference_data = [reference_data_1, reference_data_2]

            # Load test data
            test_path = 'dataset/level4/test'
            test_data = np.array(self.load_all(test_path))

        return reference_data, test_data