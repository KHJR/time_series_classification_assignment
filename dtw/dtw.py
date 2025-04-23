import numpy as np

class Dtw:
    def __init__(self):
        pass

    def cost_matrix(self, seq1, seq2):
        cost_matrix = np.empty((len(seq1), len(seq2)))

        for i in range(len(seq1)):
            for j in range(len(seq2)):
                cost_matrix[i, j] = np.linalg.norm(seq1[i] - seq2[j])

        return cost_matrix
    
    def cumulative_matrix(self, cost_matrix):
        cumulative_matrix = np.empty((len(cost_matrix) + 1, len(cost_matrix[0]) + 1))

        for i in range(len(cumulative_matrix)):
            for j in range(len(cumulative_matrix[0])):
                if i == 0 and j == 0:
                    cumulative_matrix[i, j] = 0
                elif i == 0 or j == 0:
                    cumulative_matrix[i, j] = np.inf
                else:
                    cumulative_matrix[i, j] = min(
                        cumulative_matrix[i - 1, j],
                        cumulative_matrix[i, j - 1],
                        cumulative_matrix[i - 1, j - 1]
                    ) + cost_matrix[i - 1, j - 1]

        return cumulative_matrix
    
    def distance(self, cumulative_matrix):
        return cumulative_matrix[-1, -1]
    
    def compute_dtw_distance(self, seq1, seq2):
        cost_matrix = self.cost_matrix(seq1, seq2)
        cumulative_matrix = self.cumulative_matrix(cost_matrix)
        return self.distance(cumulative_matrix)