"""Utility functions."""
import numpy as np

def threshold_matrix(matrix, threshold=0.001):
    """Sets matrix elements to zero if their absolute value is below the threshold."""
    result = np.copy(matrix)
    result[np.abs(result) < threshold] = 0
    return result