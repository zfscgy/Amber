import pickle
import numpy as np
from pathlib import Path


cifar_folder = Path(__file__).parent.joinpath("Datasets/cifar-10-python/cifar-10-batches-py")


def get_cifar_train_test():
    train_xs = []
    for i in range(1, 6):
        train_data_dict = pickle.load(cifar_folder.joinpath(f'data_batch_{i}'))
        train_xs.append(train_data_dict['data'])

    train_xs = np.vstack(train_xs)  # [50000, 3072]

    test_xs = None


gisette_folder = Path(__file__).parent.joinpath("Datasets/gisette")


def get_gisette_train_test():
    train_xs = np.loadtxt(gisette_folder.joinpath("gisette_train.data"))
    train_ys = np.loadtxt(gisette_folder.joinpath("gisette_train.labels"))
    test_xs = np.loadtxt(gisette_folder.joinpath("gisette_valid.data"))
    test_ys = np.loadtxt(gisette_folder.joinpath("gisette_valid.labels"))

    train_xs = train_xs / 500 - 1
    test_xs = test_xs / 500 - 1

    return train_xs, train_ys, test_xs, test_ys
