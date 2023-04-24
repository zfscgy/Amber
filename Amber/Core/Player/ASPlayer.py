import time

from Amber.Core.Player.Player import *
from Amber.Core.Comm.Peer import Peer


def auto_to_tensor_as(arg_indices=None):
    def inner_func(f: Callable):
        def convert_to_tensor(_self, x):
            if isinstance(x, np.ndarray) or isinstance(x, float) or isinstance(x, int):
                return _self.tfactory.local(_self.backend.encode(x))
            elif isinstance(x, Tensor):
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

        def wrapper(_self, *args, **kwargs):
            args = list(args)
            if arg_indices is None:
                for i, arg in enumerate(args):
                    args[i] = convert_to_tensor(_self, arg)
            else:
                for arg_idx in arg_indices:
                    args[arg_idx] = convert_to_tensor(_self, args[arg_idx])
            return f(_self, *args, **kwargs)
        return wrapper
    return inner_func


class ASPlayer(Player):
    def __init__(self, role: str, addr_dict: dict, triple_buffer_size: int=64,
                 backend: ASBackend = None):
        if role not in {"player0", "player1", "third-party"}:
            raise PlayerException(f"Role must be one of player0, player1 and third-party, but get {role}")
        self.role = role
        if set(addr_dict.keys()) != {"player0", "player1", "third-party"}:
            raise PlayerException(f"Address dictionary must contain player0, player1 and third-party")
        if role == "third-party":
            self.is_third_party = True
        else:
            self.is_third_party = False
            if role == "player0":
                self.other_player = "player1"
            else:
                self.other_player = "player0"

        _other_addr_dict = dict()
        for k in addr_dict:
            if k != role:
                v = addr_dict[k]
                _other_addr_dict[v] = k
        self.peer = Peer(addr_dict[role], _other_addr_dict)

        self.triple_buffer_size = triple_buffer_size
        self.triple_buffer = dict()
        self.disable_triple_buffer = False

        if backend is None:
            backend = I64ASBackend()
        if not isinstance(backend, ASBackend):
            raise PlayerException(f"ASPlayer: Must be an AS backend")
        super(ASPlayer, self).__init__(backend)

    def init_play(self):
        time.sleep(3)
        self.peer.connect_all()

        # Sync the random seed for p0 and p1
        if self.role == "player0":
            seed = self.backend.mod(self.backend.random_int(), 2**63)
            self.backend.reset_seed(seed)
            self.peer.send(self.other_player, "Init-random_seed", seed)
        elif self.role == "player1":
            seed = self.peer.recv(self.other_player, "Init-random_seed")
            self.backend.reset_seed(seed)


        # Sync random seeds for the third-party and p0, p1 for faster beaver triple generation
        if self.is_third_party:
            seed_triple_p0 = self.backend.mod(self.backend.random_int(), 2**63)
            seed_triple_p1 = self.backend.mod(self.backend.random_int(), 2**63)
            self.backend.new_rng("triple_p0", seed_triple_p0)
            self.backend.new_rng("triple_p1", seed_triple_p1)
            parallel(self.peer.send, [
                ("player0", "seed_rng_triple", seed_triple_p0),
                ("player1", "seed_rng_triple", seed_triple_p1)
            ])
        elif self.role == "player0":
            self.backend.new_rng("triple", self.peer.recv("third-party", "seed_rng_triple"))
        else:
            self.backend.new_rng("triple", self.peer.recv("third-party", "seed_rng_triple"))

    def _get_triple(self, name: str, op: Callable, shape_0: Tuple[int, ...], shape_1: Tuple[int, ...]):
        if self.disable_triple_buffer:
            triple_buffer_size = 1
        else:
            triple_buffer_size = self.triple_buffer_size

        key = f"{name}-{shape_0}-{shape_1}"
        if key not in self.triple_buffer:
            self.triple_buffer[key] = []
            """
            Generate beaver-triples
            This involves communication, so lock is used
            :return:
            """
        if len(self.triple_buffer[key]) == 0:
            if self.is_third_party:
                player1_triples = []
                for _ in range(triple_buffer_size):
                    u0 = self.backend.random_int(shape_0, "triple_p0")
                    u1 = self.backend.random_int(shape_0, "triple_p1")
                    v0 = self.backend.random_int(shape_1, "triple_p0")
                    v1 = self.backend.random_int(shape_1, "triple_p1")
                    w = op(self.backend.add(u0, u1), self.backend.add(v0, v1))
                    w0 = self.backend.random_int(self.backend.get_shape(w), "triple_p0")
                    w1 = self.backend.sub(w, w0)
                    player1_triples.append(w1)
                    self.triple_buffer[key].append(w)
                self.peer.send("player1", f"triple_w1s: {key}", player1_triples)

            if self.role == "player1":
                triples = self.peer.recv("third-party", f"triple_w1s: {key}")
                for w1 in triples:
                    u1 = self.backend.random_int(shape_0, "triple")
                    v1 = self.backend.random_int(shape_1, "triple")
                    self.triple_buffer[key].append((u1, v1, w1))

            if self.role == "player0":
                for _ in range(triple_buffer_size):
                    u0 = self.backend.random_int(shape_0, "triple")
                    v0 = self.backend.random_int(shape_1, "triple")
                    w0 = self.backend.random_int(self.backend.get_shape(op(u0, v0)), "triple")
                    self.triple_buffer[key].append((u0, v0, w0))

        return self.triple_buffer[key].pop(0)

    def new_tensor(self, tensor_getter: Callable[[], Union[int, float, np.ndarray]], *args, **kwargs):
        """
        :param tensor_getter: A function that will only executed on the owner's machine
        :param owner: player0 or player1
        :return: A shared tensor
        """
        if len(args) == 0:
            owner = kwargs.get("owner")
            if owner is None:
                owner = "player0"
        else:
            owner = args[0]

        if owner not in ["player0", "player1", "all"]:
            raise PlayerException(f"new_tensor: invalid owner {owner}")
        if owner == "all":
            tensor = self.tfactory.local(self.backend.encode(tensor_getter()))
        else:
            if self.role == owner:
                tensor = self.tfactory.shared(self.backend.encode(tensor_getter()))
                # parallel(self.peer.send, [
                #     (self.other_player, "new_tensor-shape", tensor.shape),
                #     ("third-party", "new_tensor-shape", tensor.shape)])
            else:
                # tensor_shape = self.peer.recv(owner, "new_tensor-shape")
                tensor_shape = self.backend.get_shape(self.backend.encode(tensor_getter()))
                tensor = self.tfactory.shared(self.backend.zeros(tensor_shape))
        return tensor

    def reveal(self, x: Tensor, player: str=None):
        if self.is_third_party:
            return x
        if x.type == TensorType.Local:
            return x
        if player is None:
            self.peer.send(self.other_player, "reveal-part", x.value)
            return self.tfactory.local(self.backend.add(x.value, self.peer.recv(self.other_player, "reveal-part")))
        elif player not in ["player0", "player1"]:
            raise PlayerException("reveal: can only reveal to player0 or player1")
        else:
            if self.role == player:
                other_part = self.peer.recv(self.other_player, "reveal-part")
                return self.tfactory.local(self.backend.add(other_part, x.value))
            else:
                self.peer.send(self.other_player, "reveal-part", x.value)
                return None

    def decode(self, x: Tensor):
        if x.type == TensorType.Local:
            return self.backend.decode(x.value)
        else:
            raise PlayerException("decode: cannot decode a shared tensor. Call reveal first.")

    def neg(self, tensor0: Union[Tensor, np.ndarray, float, int]):
        return self.tfactory.tensor(self.backend.neg(tensor0.value), tensor0.type)

    @auto_to_tensor_as(arg_indices=[1, 2])
    def _op_linear(self, linear_op: Callable, tensor0: Union[Tensor, np.ndarray, float, int], tensor1: [Tensor, np.ndarray, float, int]):
        if self.is_third_party:
            if tensor0.type == TensorType.Local and tensor1.type == TensorType.Local:
                new_type = TensorType.Local
            else:
                new_type = TensorType.AShared
            return self.tfactory.tensor(linear_op(tensor0.value, tensor1.value), new_type)
        if tensor0.type == TensorType.Local and tensor1.type == TensorType.Local:
            return self.tfactory.local(linear_op(tensor0.value, tensor1.value))
        elif tensor0.type == TensorType.Local:
            if self.role == "player0":
                return self.tfactory.shared(linear_op(tensor0.value, tensor1.value))
            else:
                return self.tfactory.shared(linear_op(0, tensor1.value))
        elif tensor1.type == TensorType.Local:
            if self.role == "player0":
                return self.tfactory.shared(linear_op(tensor0.value, tensor1.value))
            else:
                return self.tfactory.shared(linear_op(tensor0.value, 0))
        else:
            return self.tfactory.shared(linear_op(tensor0.value, tensor1.value))

    def add(self, tensor0: Union[Tensor, np.ndarray, float, int], tensor1: [Tensor, np.ndarray, float, int]) -> Tensor:
        return self._op_linear(self.backend.add, tensor0, tensor1)

    def sub(self, tensor0: Union[Tensor, np.ndarray, float, int], tensor1: [Tensor, np.ndarray, float, int]) -> Tensor:
        return self._op_linear(self.backend.sub, tensor0, tensor1)

    @auto_to_tensor_as(arg_indices=[1, 2])
    def _op_mul(self, mul_op: Callable,
                tensor0: Union[Tensor, np.ndarray, float, int], tensor1: [Tensor, np.ndarray, float, int]) -> Tensor:
        bitlen = self.backend.bitlen
        if tensor0.type == TensorType.Local and tensor1.type == TensorType.Local:
            return self.tfactory.local(self.backend.recode(mul_op(tensor0.value, tensor1.value)))
        if tensor0.type == TensorType.Local or tensor1.type == TensorType.Local:
            if self.is_third_party:
                return self.tfactory.shared(self.backend.zeros(self.tfactory.get_shape(mul_op(tensor0.value, tensor1.value))))
            new_val = mul_op(tensor0.value, tensor1.value)
            if self.role == "player0":
                """
                maybe_overflow_elems = self.backend.find_indices(self.backend.greater(new_val, 2 ** (bitlen - 3)))
                maybe_underflow_elems = self.backend.find_indices(self.backend.greater(- 2 ** (bitlen - 3), new_val))
                """
                maybe_overflow_elems = self.backend.greater(new_val, 2 ** (bitlen - 3)).astype(np.uint8)
                maybe_underflow_elems = self.backend.greater(- 2 ** (bitlen - 3), new_val).astype(np.uint8)
                self.peer.send(self.other_player, "overflow, underflow",
                    (self.backend.pack_bits(maybe_overflow_elems), self.backend.pack_bits(maybe_underflow_elems)))
            else:
                maybe_underflow_elems, maybe_overflow_elems = self.peer.recv(self.other_player, "overflow, underflow")
                maybe_underflow_elems = self.backend.unpack_bits(maybe_underflow_elems)
                maybe_overflow_elems = self.backend.unpack_bits(maybe_overflow_elems)

            if len(new_val.shape) != 0:
                new_overflow_vals = self.backend.sub(self.backend.select_by_indicator(new_val, maybe_overflow_elems), 2 ** (bitlen - 3))
                self.backend.set_by_indicator(new_val, maybe_overflow_elems, new_overflow_vals)
                new_underflow_vals = self.backend.add(self.backend.select_by_indicator(new_val, maybe_underflow_elems), 2 ** (bitlen - 3))
                self.backend.set_by_indicator(new_val, maybe_underflow_elems, new_underflow_vals)
            else:
                if maybe_overflow_elems == 1:
                    new_val -= 2 ** (bitlen - 3)
                if maybe_underflow_elems == 1:
                    new_val += 2 ** (bitlen - 3)

            return self.tfactory.shared(self.backend.recode(new_val))

        triple = self._get_triple(f"{mul_op.__qualname__}", mul_op, tensor0.shape, tensor1.shape)
        if self.is_third_party:
            return self.tfactory.shared(self.backend.zeros(self.backend.get_shape(triple)))
        else:
            u, v, w = triple
            x_sub_u_self = self.backend.sub(tensor0.value, u)
            y_sub_v_self = self.backend.sub(tensor1.value, v)

            if self.role == "player0":
                self.peer.send(self.other_player, "mul-x_sub_u, y_sub_v", (x_sub_u_self, y_sub_v_self))
                # The overflow elements for player1 is the underflow elements for player0, so must swap
                x_sub_u_other, y_sub_v_other, maybe_underflow_elems, maybe_overflow_elems = \
                    self.peer.recv(self.other_player, "mul-x_sub_u, y_sub_v, overflow, underflow")
                maybe_underflow_elems = self.backend.unpack_bits(maybe_underflow_elems)
                maybe_overflow_elems = self.backend.unpack_bits(maybe_overflow_elems)


                x_sub_u = self.backend.add(x_sub_u_self, x_sub_u_other)
                y_sub_v = self.backend.add(y_sub_v_self, y_sub_v_other)
                x_mul_y_self = self.backend.add(self.backend.add(mul_op(x_sub_u, v), mul_op(u, y_sub_v)), w)
                x_mul_y_self = self.backend.add(x_mul_y_self, mul_op(x_sub_u, y_sub_v))

            else:
                x_sub_u_other, y_sub_v_other = self.peer.recv(self.other_player, "mul-x_sub_u, y_sub_v")
                x_sub_u = self.backend.add(x_sub_u_self, x_sub_u_other)
                y_sub_v = self.backend.add(y_sub_v_self, y_sub_v_other)

                x_mul_y_self = self.backend.add(self.backend.add(mul_op(x_sub_u, v), mul_op(u, y_sub_v)), w)
                '''
                maybe_overflow_elems = self.backend.find_indices(
                    self.backend.greater(x_mul_y_self, 2 ** (bitlen - 3)))

                maybe_underflow_elems = self.backend.find_indices(
                    self.backend.greater(- 2 ** (bitlen - 3), x_mul_y_self))
                '''
                maybe_overflow_elems = self.backend.greater(x_mul_y_self, 2 ** (bitlen - 3))

                maybe_underflow_elems = self.backend.greater(- 2 ** (bitlen - 3), x_mul_y_self)

                self.peer.send(self.other_player, "mul-x_sub_u, y_sub_v, overflow, underflow",
                               (x_sub_u_self, y_sub_v_self,
                                self.backend.pack_bits(maybe_overflow_elems),
                                self.backend.pack_bits(maybe_underflow_elems)))


            new_underflow_vals = self.backend.add(
                self.backend.select_by_indicator(x_mul_y_self, maybe_underflow_elems), 2 ** (bitlen - 3))
            self.backend.set_by_indicator(x_mul_y_self, maybe_underflow_elems, new_underflow_vals)

            new_overflow_vals = self.backend.sub(
                self.backend.select_by_indicator(x_mul_y_self, maybe_overflow_elems), 2 ** (bitlen - 3))

            self.backend.set_by_indicator(x_mul_y_self, maybe_underflow_elems, new_underflow_vals)
            self.backend.set_by_indicator(x_mul_y_self, maybe_overflow_elems, new_overflow_vals)

            return self.tfactory.shared(self.backend.recode(x_mul_y_self))

    def mul(self, tensor0: Union[Tensor, np.ndarray, float, int], tensor1: [Tensor, np.ndarray, float, int]) -> Tensor:
        return self._op_mul(self.backend.mul, tensor0, tensor1)

    def matmul(self, tensor0: Union[Tensor, np.ndarray, float, int], tensor1: [Tensor, np.ndarray, float, int]) -> Tensor:
        return self._op_mul(self.backend.matmul, tensor0, tensor1)

    def square(self, tensor: Union[Tensor, np.ndarray]) -> Tensor:
        return self.mul(tensor, tensor)

    @auto_to_tensor_as(arg_indices=[0, 1])
    def select(self, tensor: Union[Tensor, np.ndarray], indices: Union[Tensor, np.ndarray], axis: int) -> Tensor:
        return self.tfactory.tensor(self.backend.select(tensor.value, indices.value, axis), tensor.type)

    @auto_to_tensor_as(arg_indices=[0])
    def broadcast(self, tensor: Union[Tensor, np.ndarray], shape: Tuple[int, ...]) -> Tensor:
        return self.tfactory.tensor(self.backend.broadcast(tensor.value, shape), tensor.type)

    @auto_to_tensor_as(arg_indices=[0])
    def reshape(self, tensor: Union[Tensor, np.ndarray], shape: Tuple[int, ...]) -> Tensor:
        return self.tfactory.tensor(self.backend.reshape(tensor.value, shape), tensor.type)

    @auto_to_tensor_as(arg_indices=[0])
    def transpose(self, tensor: Union[Tensor, np.ndarray], idx1: int, idx2: int) -> Tensor:
        return self.tfactory.tensor(self.backend.transpose(tensor.value, idx1, idx2), tensor.type)

    @auto_to_tensor_as(arg_indices=[0])
    def concat(self, tensors: List[Tensor], axis: int) -> Tensor:
        for tensor in tensors[1:]:
            if tensor.type != tensors[0].type:
                raise PlayerException(f"Player.concat: must be the same type. But {tensor.type} != {tensors[0].type}")
        return self.tfactory.tensor(self.backend.concat([t.value for t in tensors], axis), tensors[0].type)

    @auto_to_tensor_as(arg_indices=[0])
    def sum(self, tensor: Union[Tensor, np.ndarray, List[Tensor]], axis: Tuple[int, ...] = None) -> Tensor:
        if isinstance(tensor, List):
            for t in tensor:
                if t.type != tensor[0].type:
                    raise PlayerException(f"Player.sum: must be the same type. But {t.type} != {tensor[0].type}")
            return self.tfactory.tensor(self.backend.sum([t.value for t in tensor], axis), tensor[0].type)
        else:
            return self.tfactory.tensor(self.backend.sum(tensor.value, axis), tensor.type)
