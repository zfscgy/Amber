from Amber.Core.Base.TensorBase import *
from Amber.Core.Backend import *
from Amber.Core.Utils import *


class PlayerException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class Player:
    """
    Player is for handling tensor operation
    It must contain a backend for low-level computation
    """
    def __init__(self, backend: Backend):
        self.backend = backend
        self.tfactory = TensorFactory(backend.get_shape)

    def new_tensor(self, tensor_getter: Callable[[], Union[int, float, np.ndarray]], **kwargs):
        raise NotImplementedError()

    def neg(self, tensor0: Union[Tensor, np.ndarray, float, int]):
        raise NotImplementedError()

    def add(self, tensor0: Union[Tensor, np.ndarray, float, int], tensor1: [Tensor, np.ndarray, float, int]) -> Tensor:
        raise NotImplementedError()

    def sub(self, tensor0: Union[Tensor, np.ndarray, float, int], tensor1: [Tensor, np.ndarray, float, int]) -> Tensor:
        raise NotImplementedError()

    def mul(self, tensor0: Union[Tensor, np.ndarray, float, int], tensor1: [Tensor, np.ndarray, float, int]) -> Tensor:
        raise NotImplementedError()

    def matmul(self, tensor0: Union[Tensor, np.ndarray, float, int], tensor1: [Tensor, np.ndarray, float, int]) -> Tensor:
        raise NotImplementedError()

    def square(self, tensor: Union[Tensor, np.ndarray]) -> Tensor:
        raise NotImplementedError()

    def sigmoid(self, tensor: Union[Tensor, np.ndarray]) -> Tensor:
        raise NotImplementedError()

    def tanh(self, tensor: Union[Tensor, np.ndarray]) -> Tensor:
        raise NotImplementedError()

    def relu(self, tensor: Union[Tensor, np.ndarray], k: float) -> Tensor:
        raise NotImplementedError()

    def relu_grad(self, tensor: Union[Tensor, np.ndarray], k: float) -> Tensor:
        raise NotImplementedError()

    def select(self, tensor: Union[Tensor, np.ndarray], indices: List[int], axis: int) -> Tensor:
        raise NotImplementedError()

    def broadcast(self, tensor: Union[Tensor, np.ndarray], shape: Tuple[int, ...]) -> Tensor:
        raise NotImplementedError()

    def reshape(self, tensor: Union[Tensor, np.ndarray], shape: Tuple[int, ...]) -> Tensor:
        raise NotImplementedError()

    def transpose(self, tensor: Union[Tensor, np.ndarray], idx1: int, idx2: int) -> Tensor:
        raise NotImplementedError()

    def concat(self, tensors: List[Tensor], axis: int) -> Tensor:
        raise NotImplementedError()

    def sum(self, tensor: Union[Tensor, np.ndarray, List[Tensor]], axis: Tuple[int, ...] = None) -> Tensor:
        raise NotImplementedError()

    def mean(self, tensor: Union[Tensor, np.ndarray, List[Tensor]], axis: Tuple[int, ...] = None) -> Tensor:
        raise NotImplementedError()
