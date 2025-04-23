import matplotlib.pyplot as plt
import numpy as np

from dataloader import DataLoader

class Visualizer:
    def __init__(self):
        pass

    def visualize(self, level):
        if not 1 <= level <= 3:
            raise ValueError('Level must be between 1 and 3 for this visualization.')
        
        dataloader = DataLoader()
        reference_data, test_data = dataloader.prepare_data(level)

        if level == 1:
            self.plot_1d(reference_data, test_data, title='Data Visualization of Level 1', xlabel='Time', ylabel='Value')
        else:
            self.plot_3d(reference_data, test_data, title=f'Data Visualization of Level {level}', xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis')
        
    def plot_1d(self, reference_data, test_data, title='1D Data Plot', xlabel='X-axis', ylabel='Y-axis'):
        plt.figure(figsize=(10, 7))
        # Plot reference data
        plt.plot(reference_data[0], color='blue', label='Reference Data 1')
        plt.plot(reference_data[1], color='red', label='Reference Data 2')
        
        for i, test_seq in enumerate(test_data):
            plt.plot(test_seq, color='gray', alpha=0.6, label=f'Test Data' if i == 0 else '')
        
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.show()

    def plot_3d(self, reference_data, test_data, title='3D Data Plot', xlabel='X-axis', ylabel='Y-axis', zlabel='Z-axis'):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot reference data
        ax.plot(reference_data[0][:, 0], reference_data[0][:, 1], reference_data[0][:, 2], color='blue', label='Reference Data 1')
        ax.plot(reference_data[1][:, 0], reference_data[1][:, 1], reference_data[1][:, 2], color='red', label='Reference Data 2')
        
        for i, test_seq in enumerate(test_data):
            ax.plot(test_seq[:, 0], test_seq[:, 1], test_seq[:, 2], color='gray', alpha=0.6, label=f'Test Data' if i == 0 else '')
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    visualizer = Visualizer()
    visualizer.visualize(3)  # Change this to the desired level (1, 2, or 3)