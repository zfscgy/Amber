from enum import Enum
import numpy as np
from typing import Callable, Union

np.seterr(over='ignore')


class TensorType(Enum):
    Local = 0
    AShared = 1
    Other = 2


class Tensor:
    def __init__(self, value, shape, tensor_type: TensorType = TensorType.Local):
        self.value = value
        self.shape = shape
        self.type = tensor_type

    def __str__(self):
        return str(self.value)


class TensorFactory:
    def __init__(self, get_shape: Callable):
        self.get_shape = get_shape

    def tensor(self, x, tensor_type: TensorType):
        return Tensor(x, self.get_shape(x), tensor_type)

    def local(self, x):
        return Tensor(x, self.get_shape(x), TensorType.Local)

    def shared(self, x):
        return Tensor(x, self.get_shape(x), TensorType.AShared)


