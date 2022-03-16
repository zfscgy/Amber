import numpy as np
import pandas as pd


class DataLoader:
    def set_random_seed(self, random_seed: int):
        raise NotImplementedError

    def get_batch(self, batch_size: int):
        raise NotImplementedError()

    def shape(self):
        raise NotImplementedError()


class NPDataLoader(DataLoader):
    def __init__(self, numpy_arr: np.ndarray):
        self.data = numpy_arr
        self.random_generator = np.random.default_rng()

    def set_random_seed(self, random_seed: int):
        seed = random_seed
        self.random_generator = np.random.default_rng(seed=seed)

    def get_batch(self, batch_size: int):
        if batch_size is None:
            return self.data
        indices = self.random_generator.choice(self.data.shape[0], batch_size)
        return self.data[indices]

    def shape(self):
        return self.data.shape
