from typing import List, Callable
from Amber.NN.Operator import Node, Operator
from Amber.NN.NNOperators import Dense
from Amber.NN.OpFacotry import OpFactory
from Amber.NN.Initializers import GlorotUniform


class Layer:
    def parameters(self) -> List[Node]:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs) -> Operator:
        raise NotImplementedError()


class DenseLayer(Layer):
    def __init__(self, in_dim: int, out_dim: int, initializer: Callable = None, op_factory: OpFactory=None, owner: str=None):
        if initializer is None:
            initializer = GlorotUniform(in_dim, out_dim)
        if op_factory is None:
            op_factory = OpFactory.default
        self.op_factory = op_factory
        init_w, init_b = initializer()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w = op_factory.new_node(init_w, owner=owner)
        self.b = op_factory.new_node(init_b, owner=owner)

    def parameters(self) -> List[Node]:
        return [self.w, self.b]

    def __call__(self, x: Node):
        return self.op_factory.dense(x, self.w, self.b, self.in_dim, self.out_dim)
