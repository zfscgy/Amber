from typing import List, Union
import numpy as np


class Initializer:
    def __call__(self) -> Union[np.ndarray, List[np.ndarray]]:
        raise NotImplementedError()


class DenseInitializer(Initializer):
    def __init__(self, in_dim: int, out_dim: int):
        self.in_dim = in_dim
        self.out_dim = out_dim


class GlorotUniform(DenseInitializer):
    def __call__(self):
        r = np.sqrt(6 / (self.in_dim + self.out_dim))
        return np.random.uniform(-r, r, [self.in_dim, self.out_dim]), np.zeros([self.out_dim])
