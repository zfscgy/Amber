from Amber.Core.Backend.Backend import *
from Amber.Core.Backend.NumpyBackend import NumpyBackend


class I64ASBackend(ASBackend, NumpyBackend):
    def __init__(self, precision: int = 21, seed: int = np.random.randint(0, 2**63-1)):
        self.precision = precision
        self.bitlen = 64
        NumpyBackend.__init__(self, seed, np.int64)

    def encode(self, x: Union[np.ndarray, float, int]):
        return np.array(x * 2 ** self.precision).astype(np.int64)

    def decode(self, x):
        return x.astype(np.float64) / (2 ** self.precision)

    def recode(self, x):
        q, r = np.divmod(x, 2 ** self.precision)
        q = np.array(q)
        r = np.array(r)
        q[r >= 2 ** (self.precision - 1)] += 1
        return q

    def sigmoid(self, x):
        return self.encode(1 / 1 + np.exp(- self.decode(x)))

    def tanh(self, x):
        return self.encode(np.tanh(self.decode(x)))

    def square(self, x):
        return self.mul(x, x)


class I64RTASBackend(RTASBackend, I64ASBackend):
    def __init__(self, precision: int = 21, seed: int = np.random.randint(0, 2**63-1)):
        I64ASBackend.__init__(self, precision, seed)
        self.permutation_generator = np.random.default_rng(seed)

    def reset_seed(self, seed: int, name: str=None):
        I64ASBackend.reset_seed(self, seed)
        if name is None:
            self.permutation_generator = np.random.default_rng(seed)

    def random_permutation(self, size):
        permutation = self.permutation_generator.permutation(size)
        inv_permutation = np.zeros_like(permutation)
        arange = np.arange(size)
        inv_permutation[permutation] = arange
        return permutation, inv_permutation
