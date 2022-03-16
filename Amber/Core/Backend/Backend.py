import numpy as np
from builtins import slice
from typing import Callable, List, Tuple, Union


class Backend:
    def new_rng(self, name: str, seed: int):
        raise NotImplementedError()

    def encode(self, x: Union[np.ndarray, float, int]):
        raise NotImplementedError()

    def decode(self, x):
        raise NotImplementedError()

    def random_int(self, shape: Tuple[int] = None, name: str = None):
        raise NotImplementedError()

    def random_int_range(self, shape: Tuple[int, ...], low: int, high: int, name: str):
        raise NotImplementedError()

    def reset_seed(self, seed: int, name: str=None):
        raise NotImplementedError()

    def get_shape(self, x):
        raise NotImplementedError()

    def equal(self, x, y):
        raise NotImplementedError()

    def greater(self, x, y):
        raise NotImplementedError()

    def find_indices(self, x):
        raise NotImplementedError()

    def set_by_indicator(self, x: np.ndarray, indicator: np.ndarray, values: np.ndarray):
        raise NotImplementedError()

    def set_by_indices(self, x, indices, values):
        raise NotImplementedError()

    def zeros(self, shape: Tuple[int, ...]):
        raise NotImplementedError()

    def neg(self, x):
        raise NotImplementedError()

    def add(self, x, y):
        raise NotImplementedError()

    def sub(self, x, y):
        raise NotImplementedError()

    def mul(self, x, y):
        raise NotImplementedError()

    def matmul(self, x, y):
        raise NotImplementedError()

    def mod(self, x, y):
        raise NotImplementedError()

    def square(self, x):
        raise NotImplementedError()

    def sigmoid(self, x):
        raise NotImplementedError()

    def tanh(self, x):
        raise NotImplementedError()

    def relu(self, x: np.ndarray, k: float):
        raise NotImplementedError()

    def relu_grad(self, x:  np.ndarray, k: float):
        raise NotImplementedError()

    def select(self, x, indices: List[int], axis: int):
        raise NotImplementedError()

    def select_by_indicator(self, x: np.ndarray, indicator: np.ndarray):
        raise NotImplementedError()

    def select_by_indices(self, x: np.ndarray, indices: np.ndarray):
        raise NotImplementedError()

    def select_slices(self, x, slices: Tuple[slice, ...]):
        raise NotImplementedError()

    def broadcast(self, x, shape: Tuple[int, ...]):
        raise NotImplementedError()

    def reshape(self, x, shape: Tuple[int, ...]):
        raise NotImplementedError()

    def transpose(self, x, axis0: int, axis1: int):
        """
        This is the same as np.swapaxes
        :param x:
        :param axis0:
        :param axis1:
        :return:
        """
        return NotImplementedError()

    def sum(self, xs, axis: Tuple[int, ...]):
        """
        This is the same as np.sum with keepdims=True for convenience

        :param xs:
        :param axis:
        :return:
        """
        raise NotImplementedError()

    def mean(self, xs, axis: Tuple[int, ...]):
        raise NotImplementedError()

    def concat(self, xs, axis: int):
        raise NotImplementedError()


class ASBackend(Backend):
    def __init__(self):
        self.bitlen = 0
        self.precision = 0

    def random_triple(self, shape_0: Tuple[int, ...], shape_1: Tuple[int, ...], mul_op: Callable):
        raise NotImplementedError()

    def recode(self, x):
        raise NotImplementedError()


class RTASBackend(ASBackend):
    def random_permutation(self, size):
        raise NotImplementedError()
