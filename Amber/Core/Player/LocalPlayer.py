from Amber.Core.Player.Player import *


def auto_to_tensor(arg_indices=None):
    def inner_func(f: Callable):
        def convert_to_tensor(_self, x):
            if isinstance(x, np.ndarray) or isinstance(x, float) or isinstance(x, int):
                return _self.tfactory.local(_self.backend.encode(x))
            elif isinstance(x, Tensor):
                if x.type != TensorType.Local:
                    raise PlayerException(f"convert_to_tensor: Tensor type can only be Local but get {x.type.name}")
                return x
            # A list of tensors
            elif isinstance(x, List):
                for i, v in enumerate(x):
                    if isinstance(v, Tensor):
                        pass
                    elif isinstance(v, np.ndarray):
                        x[i] = _self.tfactory.local(_self.backend.encode(x))
                    else:
                        PlayerException(f"convert_to_tensor: cannot convert type {x.__class__} to tensor")
            else:
                raise PlayerException(f"convert_to_tensor: cannot convert type {x.__class__} to tensor")

        def wrapper(_self, *args):
            args = list(args)
            if arg_indices is None:
                for i, arg in enumerate(args):
                    args[i] = convert_to_tensor(_self, arg)
            else:
                for arg_idx in arg_indices:
                    args[arg_idx] = convert_to_tensor(_self, args[arg_idx])
            return _self.tfactory.local(f(_self, *args))
        return wrapper
    return inner_func


class LocalPlayer(Player):
    def __init__(self, backend: NumpyBackend = None):
        if backend is None:
            backend = NumpyBackend()
        super(LocalPlayer, self).__init__(backend)

    def new_tensor(self, tensor_getter: Callable[[], Union[int, float, np.ndarray]], **kwargs):
        """
        :param tensor_getter: A function that will only executed on the owner's machine
        :param owner: player0 or player1
        :return: A shared tensor
        """
        tensor = self.tfactory.local(np.array(tensor_getter()))
        return tensor

    @auto_to_tensor()
    def neg(self, tensor0: Union[Tensor, np.ndarray, float, int]):
        return self.backend.neg(tensor0.value)

    @auto_to_tensor(arg_indices=[0, 1])
    def add(self, tensor0: Union[Tensor, np.ndarray, float, int], tensor1: [Tensor, np.ndarray, float, int]) -> Tensor:
        return self.backend.add(tensor0.value, tensor1.value)

    @auto_to_tensor(arg_indices=[0, 1])
    def sub(self, tensor0: Union[Tensor, np.ndarray, float, int], tensor1: [Tensor, np.ndarray, float, int]) -> Tensor:
        return self.backend.sub(tensor0.value, tensor1.value)

    @auto_to_tensor(arg_indices=[0, 1])
    def mul(self, tensor0: Union[Tensor, np.ndarray, float, int], tensor1: [Tensor, np.ndarray, float, int]) -> Tensor:
        return self.backend.mul(tensor0.value, tensor1.value)

    @auto_to_tensor(arg_indices=[0, 1])
    def matmul(self, tensor0: Union[Tensor, np.ndarray, float, int], tensor1: [Tensor, np.ndarray, float, int]) -> Tensor:
        return self.backend.matmul(tensor0.value, tensor1.value)

    @auto_to_tensor()
    def square(self, tensor: Union[Tensor, np.ndarray]) -> Tensor:
        return self.backend.square(tensor.value)

    @auto_to_tensor()
    def sigmoid(self, tensor: Union[Tensor, np.ndarray]) -> Tensor:
        return self.backend.sigmoid(tensor.value)

    @auto_to_tensor()
    def tanh(self, tensor: Union[Tensor, np.ndarray]) -> Tensor:
        return self.backend.tanh(tensor.value)

    @auto_to_tensor(arg_indices=[0])
    def relu(self, tensor: Union[Tensor, np.ndarray], k: float) -> Tensor:
        return self.backend.relu(tensor.value, k)

    @auto_to_tensor(arg_indices=[0])
    def relu_grad(self, tensor: Union[Tensor, np.ndarray], k: float) -> Tensor:
        return self.backend.relu_grad(tensor.value, k)

    @auto_to_tensor(arg_indices=[0, 1])
    def select(self, tensor: Union[Tensor, np.ndarray], indices: Union[Tensor, np.ndarray], axis: int) -> Tensor:
        return self.backend.select(tensor.value, indices.value, axis)

    @auto_to_tensor(arg_indices=[0])
    def broadcast(self, tensor: Union[Tensor, np.ndarray], shape: Tuple[int, ...]) -> Tensor:
        return self.backend.broadcast(tensor.value, shape)

    @auto_to_tensor(arg_indices=[0])
    def reshape(self, tensor: Union[Tensor, np.ndarray], shape: Tuple[int, ...]) -> Tensor:
        return self.backend.reshape(tensor.value, shape)

    @auto_to_tensor(arg_indices=[0])
    def transpose(self, tensor: Union[Tensor, np.ndarray], idx1: int, idx2: int) -> Tensor:
        return self.backend.transpose(tensor.value, idx1, idx2)

    @auto_to_tensor(arg_indices=[0])
    def concat(self, tensors: List[Tensor], axis: int) -> Tensor:
        return self.backend.concat([t.value for t in tensors], axis)

    @auto_to_tensor(arg_indices=[0])
    def sum(self, tensor: Union[Tensor, np.ndarray, List[Tensor]], axis: Tuple[int, ...] = None) -> Tensor:
        return self.backend.sum(tensor.value, axis)

    @auto_to_tensor()
    def mean(self, tensor: Union[Tensor, np.ndarray, List[Tensor]], axis: Tuple[int, ...] = None) -> Tensor:
        return self.backend.mean(tensor.value, axis)


