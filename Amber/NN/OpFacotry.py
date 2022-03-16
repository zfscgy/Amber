from typing import Callable
from Amber.NN.NNOperators import *
from Amber.NN.Graph import *


def new_operator(input_ops=None):
    def inner_func(f: Callable):
        def wrapper(_opf, *args, **kwargs):
            out_op = f(_opf, *args, **kwargs)
            args = list(args)
            if input_ops is None:
                pass
            else:
                args = [args[i] for i in input_ops]
            for arg in args:
                if out_op not in _opf.dependency_dict:
                    _opf.dependency_dict[out_op] = []
                else:
                    _opf.dependency_dict[out_op].append(arg)
            out_op.forward()
            return out_op
        return wrapper
    return inner_func



class OpFactory:
    default = None

    def __init__(self, player: Player):
        self.player = player
        self.dependency_dict = dict()

    def clear_graph(self):
        self.dependency_dict.clear()

    def new_node(self, x, **kwargs):
        if callable(x):
            return Node(self.player.new_tensor(x, **kwargs))
        else:
            return Node(self.player.new_tensor(lambda: x, **kwargs))

    @new_operator()
    def identity(self, x: Node) -> Operator:
        return Identity(self.player, [x])

    @new_operator()
    def negative(self, x: Node) -> Operator:
        return Negative(self.player, [x])

    @new_operator()
    def add(self, x: Node, y: Node) -> Operator:
        return Add(self.player, [x, y])

    @new_operator()
    def sub(self, x: Node, y: Node) -> Operator:
        return Sub(self.player, [x, y])

    @new_operator()
    def mul(self, x: Node, y: Node) -> Operator:
        return Mul(self.player, [x, y])

    @new_operator()
    def matmul(self, x: Node, y: Node) -> Operator:
        return Matmul(self.player, [x, y])

    @new_operator(input_ops=[0, 1])
    def broadcast(self, x: Node, y: Node, repeat_axes: List[int], n_heading_axes: int = 0) -> Operator:
        return Broadcast(self.player, [x], repeat_axes, y, n_heading_axes)

    @new_operator(input_ops=[0])
    def reshape(self, x: Node, original_shape: Tuple[int, ...], new_shape: Tuple[int, ...]):
        return Reshape(self.player, [x], original_shape, new_shape)

    @new_operator(input_ops=[0])
    def sum(self, x: Node, axis: List[int]= None, n_heading_axes: int = 0) -> Operator:
        return Sum(self.player, [x], axis, n_heading_axes)

    @new_operator(input_ops=[0])
    def mean(self, x: Node, axis: List[int] = None, n_heading_axes: int = 0):
        return Mean(self.player, [x], axis, n_heading_axes)

    @new_operator(input_ops=[0])
    def transpose(self, x: Node, axis0: int = -1, axis1: int = -2) -> Operator:
        return Transpose(self.player, [x], axis0, axis1)

    @new_operator()
    def square(self, x: Node) -> Operator:
        return Square(self.player, [x])

    @new_operator()
    def sigmoid(self, x: Node) -> Operator:
        return Sigmoid(self.player, [x])

    @new_operator()
    def tanh(self, x: Node) -> Operator:
        return Tanh(self.player, [x])

    @new_operator(input_ops=[0])
    def relu(self, x: Node, k: float):
        return Relu(self.player, [x], k)

    @new_operator()
    def relu_grad(self, x: Node, k: float):
        return ReluGrad(self.player, [x], k)

    @new_operator(input_ops=[0])
    def mean_square_error(self, x: Node, y: Node, dim: int):
        return MSEError(self.player, [x, y], dim)

    @new_operator(input_ops=[0, 1, 2])
    def dense(self, x: Node, w: Node, b: Node, in_dim: int, out_dim: int):
        return Dense(self.player, [x, w, b], in_dim, out_dim)

    def gradient_on(self, output_op: Operator, params: List[Node], scale: float=None) -> List[Node]:
        if scale is None:
            scale_node = self.player.new_tensor(lambda: 1, owner='all')
        else:
            scale_node = self.player.new_tensor(lambda: scale, owner='all')
        input_nodes = []
        operators = [output_op]

        remain_ops = [output_op]
        while len(remain_ops) != 0:
            next_op = remain_ops.pop()
            for node in next_op.input_nodes:
                if isinstance(node, Operator):
                    remain_ops.insert(0, node)
                    if node not in operators:
                        operators.append(node)
                else:
                    if node not in params and node not in input_nodes:
                        input_nodes.append(node)

        operators = list(reversed(operators))
        graph = Graph(self.player, input_nodes, params, operators)
        grads = graph.compute_gradients(scale_node)
        return grads

    def use_as_default(self):
        OpFactory.default = self
        Node.__neg__ = lambda x: self.negative(x)
        Node.__add__ = lambda x, y: self.add(x, y)
        Node.__sub__ = lambda x, y: self.sub(x, y)
        Node.__mul__ = lambda x, y: self.mul(x, y)
        Node.__matmul__ = lambda x, y: self.matmul(x, y)
