import numpy as np
from Amber.Core.Backend import I64RTASBackend


backend = I64RTASBackend()


def test_permutation():
    arr = np.random.randint(0, 100, 10)
    random_perm, inv_perm = backend.random_permutation(10)
    assert np.allclose(arr[random_perm][inv_perm], arr)


if __name__ == '__main__':
    test_permutation()
