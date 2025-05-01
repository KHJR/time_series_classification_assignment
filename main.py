from dtw.dtw_classifier import DtwClassifier
from gru.gru_trainer import GruTrainer
from gru.gru_classifier import GruClassifier

class Main:
    def __init__(self):
        pass

    def train_gru(self, level, num_epochs=200, verbose=True, save=True, test=False):
        gruTrainer = GruTrainer(level=level)
        gruTrainer.train(num_epochs=num_epochs, verbose=verbose)

        if save: gruTrainer.save_model()
        if test: main_instance.classify(level, mode='gru')

        print(f'GRU model trained for level {level}.')

    def classify(self, level, mode='dtw'):
        if mode == 'dtw':
            dtwClassifier = DtwClassifier()
            dtw_results = dtwClassifier.classify_level(level)
            print(f'DTW Classifier Results for Level {level}: {dtw_results}')
        elif mode == 'gru':
            gruClassifier = GruClassifier(model=f'models/gru_model_level_{level}.pth')
            gru_results = gruClassifier.classify_level(level)
            print(f'GRU Classifier Results for Level {level}: {gru_results}')
        else:
            raise ValueError('Invalid mode. Choose "dtw" or "gru".')
    
    def classify_all_levels(self, mode='dtw'):
        for level in range(1, 5):
            self.classify(level, mode=mode)


if __name__ == '__main__':
    main_instance = Main()
    # level = 1  # Change this to the desired level (1, 2, 3, or 4)

    # Uncomment the following line to train the GRU model
    # main_instance.train_gru(level)

    main_instance.classify_all_levels(mode='dtw')

    # The code for visualizing the data is in the utils/visualizer.py file.