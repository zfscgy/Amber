import numpy as np


def array_close(x: np.ndarray, y: np.ndarray):
    return np.allclose(x, y, 1e-2, 1e-2) or np.max(np.abs(x - y)) < 1e-3
