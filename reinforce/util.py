import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def tabular_iter(qmat: np.ndarray):
    for s in range(qmat.shape[0]):
        for a in range(qmat.shape[1]):
            pass

