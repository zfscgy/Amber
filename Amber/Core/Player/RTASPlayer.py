from Amber.Core.Player.ASPlayer import *


class RTASPlayer(ASPlayer):
    def __init__(self, role: str, addr_dict: dict, triple_buffer_size: int=512,
                 rtas_backend: RTASBackend = None, float_backend: Backend = None):
        if rtas_backend is None:
            rtas_backend = I64RTASBackend(21)
        super(RTASPlayer, self).__init__(role, addr_dict, triple_buffer_size, rtas_backend)
        if float_backend is None:
            float_backend = NumpyBackend()
        self.float_backend = float_backend

    def init_play(self):
        super(RTASPlayer, self).init_play()

        # sync seeds for element-wise functions
        if self.role == "third-party":
            seed_ewf = self.backend.mod(self.backend.random_int(), 2**63)
            self.backend.new_rng("ewf", seed_ewf)
            self.peer.send("player1", "seed_ewf", seed_ewf)
        elif self.role == "player1":
            self.backend.new_rng("ewf", self.peer.recv("third-party", "seed_ewf"))

    @auto_to_tensor_as(arg_indices=[0])
    def _element_wise(self, tensor: Union[Tensor, np.ndarray], op: Callable, **kwargs) -> Tensor:
        if tensor.type == TensorType.Local:
            return self.tfactory.local(self.backend.sigmoid(tensor.value))
        elif tensor.type == TensorType.AShared:
            shape = tensor.shape
            if self.role in ["player0", "player1"]:
                flattened_size = int(np.prod(shape))
                random_perm, inv_perm = self.backend.random_permutation(flattened_size)
                permuted_tensor = self.backend.select(
                    self.backend.reshape(tensor.value, (flattened_size,)), random_perm, 0)
                self.peer.send("third-party", "permuted_tensor", permuted_tensor)
            else:
                res, err = parallel(self.peer.recv, [("player0", "permuted_tensor"), ("player1", "permuted_tensor")])
                decoded_input = self.backend.decode(self.backend.add(res[0], res[1]))
                decoded_output = op(decoded_input, **kwargs)
                encoded_output = self.backend.encode(decoded_output)
                share1 = self.backend.random_int(self.tfactory.get_shape(encoded_output), "ewf")
                share0 = self.backend.sub(encoded_output, share1)
                self.peer.send("player0", "permuted_result", share0)
                return self.tfactory.shared(self.backend.zeros(shape))

            if self.role == "player0":
                output_share = self.peer.recv("third-party", "permuted_result")
                return self.tfactory.shared(self.backend.reshape(self.backend.select(output_share, inv_perm, 0), shape))
            else:
                return self.tfactory.shared(self.backend.reshape(
                    self.backend.select(self.backend.random_int(flattened_size, "ewf"), inv_perm, 0), shape))

    def sigmoid(self, tensor: Union[Tensor, np.ndarray]) -> Tensor:
        return self._element_wise(tensor, self.float_backend.sigmoid)

    def tanh(self, tensor: Union[Tensor, np.ndarray]) -> Tensor:
        return self._element_wise(tensor, self.float_backend.tanh)

    def relu(self, tensor: Union[Tensor, np.ndarray], k: float) -> Tensor:
        return self._element_wise(tensor, self.float_backend.relu, k=k)

    def relu_grad(self, tensor: Union[Tensor, np.ndarray], k: float) -> Tensor:
        return self._element_wise(tensor, self.float_backend.relu_grad, k=k)
