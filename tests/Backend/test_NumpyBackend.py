from Amber.Core.Backend import NumpyBackend


backend = NumpyBackend()


def test_add():
    backend.add(1, 2)


def test_matmul():
    backend.matmul([[1, 2]], [[3], [4]])


if __name__ == '__main__':
    test_add()
    test_matmul()
