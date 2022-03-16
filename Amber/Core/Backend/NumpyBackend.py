import numpy as np

from Amber.Core.Backend.Backend import *


class NumpyBackend(Backend):
    def __init__(self, seed: int = np.random.randint(0, 2**63-1), dtype=np.float):
        self.default_rng = np.random.default_rng(seed)
        self.dtype = dtype
        self.rngs = dict()

    def new_rng(self, name: str, seed: int):
        self.rngs[name] = np.random.default_rng(seed)

    def encode(self, x: Union[np.ndarray, float, int]):
        return np.array(x)

    def decode(self, x):
        return x

    def pack_bits(self, x: np.ndarray):
        return np.packbits(x), x.shape

    def unpack_bits(self, packed: tuple):
        bits, shape = packed
        # The bits are multiples of 8, hence needs to truncate to remove extra data.
        arr = np.unpackbits(bits)[:np.prod(shape)]
        return arr.reshape(shape).astype(np.bool)

    def random_int(self, shape: Tuple[int, ...] = None, name: str=None):
        if not name:
            generator = self.default_rng
        else:
            generator = self.rngs[name]
        return generator.integers(-2 ** 63, 2 ** 63 - 1, shape, dtype=np.int64)

    def random_int_range(self, shape: Tuple[int, ...], low: int, high: int, name: str=None):
        if not name:
            generator = self.default_rng
        else:
            generator = self.rngs[name]
        return generator.integers(low, high, shape)

    def reset_seed(self, seed: int, name: str=None):
        if not name:
            self.default_rng = np.random.default_rng(seed)
        else:
            self.rngs[name] = np.random.default_rng(seed)

    def get_shape(self, x: np.ndarray):
        return x.shape

    def equal(self, x: np.ndarray, y: np.ndarray):
        return x == y

    def greater(self, x: np.ndarray, y: np.ndarray):
        return x > y

    def find_indices(self, x: np.ndarray):
        return np.argwhere(x)

    def set_by_indices(self, x: np.ndarray, indices: np.ndarray, values: np.ndarray):
        if len(indices) == 0:
            return
        slices = []
        for i in range(indices.shape[1]):
            slices.append(indices[:, i])
        x[tuple(slices)] = values

    def set_by_indicator(self, x: np.ndarray, indicator: np.ndarray, values: np.ndarray):
        x[indicator > 0] = values

    def select(self, x: np.ndarray, indices: np.ndarray, axis: int):
        return np.take(x, indices, axis)

    def select_by_indicator(self, x: np.ndarray, indicator: np.ndarray):
        return x[indicator]

    def select_by_indices(self, x: np.ndarray, indices: np.ndarray):
        if len(indices) == 0:
            return []
        slices = []
        for i in range(indices.shape[1]):
            slices.append(indices[:, i])
        return x[tuple(slices)]

    def select_slices(self, x: np.ndarray, slices: Tuple[slice, ...]):
        return x[slices]

    def zeros(self, shape: Tuple[int, ...]):
        return np.zeros(shape).astype(self.dtype)

    def neg(self, x: np.ndarray):
        return -x

    def add(self, x: np.ndarray, y: np.ndarray):
        return np.add(x, y)

    def sub(self, x: np.ndarray, y: np.ndarray):
        return np.subtract(x, y)

    def mul(self, x: np.ndarray, y: np.ndarray):
        return np.array(np.multiply(x, y))

    def matmul(self, x: np.ndarray, y: np.ndarray):
        return np.matmul(x, y)

    def mod(self, x: np.ndarray, y: np.ndarray):
        return np.divmod(x, y)[1].astype(self.dtype)

    def square(self, x: np.ndarray):
        return np.square(x)

    def sigmoid(self, x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x: np.ndarray):
        return np.tanh(x)

    def relu(self, x: np.ndarray, k: float):
        y = np.copy(x)
        y[x >= 0] = x[x >= 0]
        y[x < 0] = k * x[x < 0]
        return y

    def relu_grad(self, x: np.ndarray, k: float):
        y = np.copy(x)
        y[x >= 0] = 1
        y[x < 0] = k
        return y

    def broadcast(self, x: np.ndarray, shape: Tuple[int, ...]):
        return np.broadcast_to(x, shape)

    def reshape(self, x, shape: Tuple[int, ...]):
        return np.reshape(x, shape)

    def transpose(self, x, axis0: int, axis1: int):
        return np.swapaxes(x, axis0, axis1)

    def sum(self, xs, axis: Tuple[int, ...]):
        return np.sum(xs, axis, keepdims=True)

    def mean(self, xs, axis: Tuple[int, ...]):
        return np.mean(xs, axis)

    def concat(self, xs, axis: int):
        return np.concatenate(xs, axis)
