from typing import List, Tuple, Union
import numpy as np
from Amber.Core.Base import Tensor
from Amber.Core.Player import Player


class OpException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


def check_op_input(n_innodes: int):
    def inner_func(op_func):
        def wrapper(_self, _player, input_nodes, *args, **kwargs):
            if len(input_nodes) != n_innodes:
                raise OpException(f"Operator-forward: number of input nodes must be {n_innodes} but is {len(input_nodes)}")
            return op_func(_self, _player, input_nodes, *args, **kwargs)
        return wrapper
    return inner_func


class Node:
    """
    A container to contain tensors. It is somewhat like the tf.Variable or torch.Variable.
    Nodes are fixed in an Operator, but its tensor can be changed.
    """
    def __init__(self, tensor: Tensor = None):
        self.tensor = tensor

    def set(self, tensor: Union[Tensor, None]):
        self.tensor = tensor

    def get(self) -> Union[Tensor, None]:
        return self.tensor


class Operator(Node):
    """
    An operator connects several nodes as inputs, and has an output node
    When forward is called, a specific computation is performed and the output node's tensor is set to be the result
    It also has a gradients method to compute the operator of the gradients on inputs
    """
    def __init__(self, player: Player, input_nodes: List[Node], name: str=None):
        super(Operator, self).__init__()
        self.player = player
        self.name = name
        self.input_nodes = input_nodes
        self.tensor = None

    def forward(self):
        raise NotImplementedError()

    def get(self) -> Union[Tensor, None]:
        if self.tensor is None:
            self.forward()
        return super(Operator, self).get()

    def clear_cache(self):
        self.set(None)

    def gradient(self, output_grad: Node) -> List:
        """
        :return: A list of operators represents the gradients of this operators' inputs.
        The gradient operator takes two inputs:
         - First is the gradient of this operator;
         - Second is all the other inputs of this operator
        """
        raise NotImplementedError()


class Identity(Operator):
    @check_op_input(n_innodes=1)
    def __init__(self, player: Player, nodes: List[Node], name: str = "Identity"):
        super(Identity, self).__init__(player, nodes, name)

    def forward(self):
        self.set(self.input_nodes[0].get())

    def gradient(self, output_grad: Node) -> List:
        raise [Identity(self.player, [output_grad], name=f"grad_{self.name}")]


class Negative(Operator):
    @check_op_input(n_innodes=1)
    def __init__(self, player: Player, nodes: List[Node], name: str = "Negative"):
        super(Negative, self).__init__(player, nodes, name)

    def forward(self):
        self.set(self.player.neg(self.input_nodes[0].get()))

    def gradient(self, output_grad: Node) -> List:
        raise [Negative(self.player, [output_grad], name=f"grad_{self.name}")]


class Add(Operator):
    @check_op_input(n_innodes=2)
    def __init__(self, player: Player, nodes: List[Node], name: str = "Add"):
        super(Add, self).__init__(player, nodes, name)

    def forward(self):
        self.set(self.player.add(self.input_nodes[0].get(), self.input_nodes[1].get()))

    def gradient(self, output_grad: Node) -> List[Operator]:
        return [Identity(self.player, [output_grad], name=f"grad_{self.name}_innode0"),
                Identity(self.player, [output_grad], name=f"grad_{self.name}_innode1")]


class Sub(Operator):
    @check_op_input(n_innodes=2)
    def __init__(self, player: Player, nodes: List[Node], name: str = "Sub"):
        super(Sub, self).__init__(player, nodes, name)

    def forward(self):
        self.set(self.player.sub(self.input_nodes[0].get(), self.input_nodes[1].get()))

    def gradient(self, output_grad: Node) -> List[Operator]:
        return [Identity(self.player, [output_grad], name=f"grad_{self.name}_innode0"),
                Negative(self.player, [output_grad], name=f"grad_{self.name}_innode1")]


class Mul(Operator):
    @check_op_input(n_innodes=2)
    def __init__(self, player: Player, nodes: List[Node], name: str = "Mul"):
        super(Mul, self).__init__(player, nodes, name)

    def forward(self):
        self.set(self.player.mul(self.input_nodes[0].get(), self.input_nodes[1].get()))

    def gradient(self, output_grad: Node) -> List:
        return [Mul(self.player, [output_grad, self.input_nodes[1]], f"grad_{self.name}_innode0"),
                Mul(self.player, [output_grad, self.input_nodes[0]], f"grad_{self.name}_innode1")]


class RefOperator(Operator):
    def __init__(self, player: Player, input_nodes: List[Node], ref_node: Node, name: str):
        super(RefOperator, self).__init__(player, input_nodes, name)
        self.ref_node = ref_node


class Broadcast(RefOperator):
    @check_op_input(n_innodes=1)
    def __init__(self, player: Player, input_nodes: List[Node],
                 repeat_axes: List[int], ref_node: Node, n_heading_axes: int = 0, name: str = "Broadcast"):
        super(Broadcast, self).__init__(player, input_nodes, ref_node, name)
        self.repeat_axes = repeat_axes
        self.n_heading_axes = n_heading_axes

    def forward(self):
        input_shape = self.input_nodes[0].get().shape
        for repeat_axis in self.repeat_axes:
            if repeat_axis >= len(input_shape):
                raise OpException(f"Broadcast.forward: axis {repeat_axis} is greater than input dimension")
            if self.input_nodes[0].get().shape[repeat_axis] != 1:
                raise OpException(f"Broadcast.forward: illegal input shape {self.input_nodes[0].get().shape}")
        if len(input_shape) + self.n_heading_axes != len(self.ref_node.get().shape):
            raise OpException(f"Broadcast.forward: ref_node do not match the axes")
        self.set(self.player.broadcast(self.input_nodes[0].get(), self.ref_node.get().shape))

    def gradient(self, output_grad: Node) -> List[Operator]:
        def recalc(axis):
            if axis >= 0:
                return self.n_heading_axes + axis
            else:
                return axis

        sum_axes = [recalc(repeat_axis) for repeat_axis in self.repeat_axes]
        return [
            Sum(self.player, [output_grad], list(range(self.n_heading_axes)) + sum_axes, self.n_heading_axes,
                f"grad_{self.name}")
        ]


class Sum(Operator):
    @check_op_input(n_innodes=1)
    def __init__(self, player: Player, input_nodes: List[Node],
                 sum_axes: List[int] = None, n_heading_axes: int = 0, name: str = "Sum"):
        super(Sum, self).__init__(player, input_nodes, name)
        if sum_axes is None:
            sum_axes = []
        self.sum_axes = sum_axes
        self.n_heading_axes = n_heading_axes

    def forward(self):
        input_shape = self.input_nodes[0].get().shape
        new_shape = list(input_shape).copy()
        for axis in self.sum_axes:
            new_shape[axis] = 1
            if axis >= len(input_shape):
                raise OpException(f"Sum.forward: axis {axis} is greater than input dimension")
        if self.n_heading_axes > len(input_shape):
            raise OpException(f"Sum.forward: number of heading axes is greater than input dimension")
        new_shape = new_shape[self.n_heading_axes:]
        self.set(
            self.player.reshape(self.player.sum(self.input_nodes[0].get(), tuple(self.sum_axes)), tuple(new_shape))
        )

    def gradient(self, output_grad: Node) -> List[Operator]:
        return [
            Broadcast(self.player, [output_grad], self.sum_axes[self.n_heading_axes:],
                      self.input_nodes[0], self.n_heading_axes)
        ]


class Transpose(Operator):
    @check_op_input(n_innodes=1)
    def __init__(self, player: Player, input_nodes: List[Node], axis0: int = -1, axis1: int = -2,
                 name: str = "Transpose"):
        super(Transpose, self).__init__(player, input_nodes, name)
        self.axis0 = axis0
        self.axis1 = axis1

    def forward(self):
        self.set(self.player.transpose(self.input_nodes[0].get(), self.axis0, self.axis1))

    def gradient(self, output_grad: Node) -> List[Operator]:
        return [Transpose(self.player, [output_grad], self.axis1, self.axis0, f"grad_{self.name}")]


class Size(Operator):
    def __init__(self, player: Player, ref_node: Node, axis: Union[List[int], int] = None, name: str = "Size",
                 reciprocal: bool = False):
        super(Size, self).__init__(player, [], name)
        self.ref_node = ref_node
        if isinstance(axis, int):
            axis = [axis]
        self.axis = axis
        self.reciprocal = reciprocal

    def forward(self):
        if self.ref_node.get() is None:
            raise OpException("Size.forward: ref_node must be computed first")
        shape = self.ref_node.get().shape
        size = 1
        if self.axis is None:
            for dim in shape:
                size *= dim
        else:
            for idx in self.axis:
                if idx >= len(shape):
                    raise OpException(f"Size.forward: cannot get axis {idx} when shape is {shape}")
                else:
                    size *= shape[idx]
        if not self.reciprocal:
            self.set(self.player.new_tensor(lambda: np.array(size), owner="all"))
        else:
            self.set(self.player.new_tensor(lambda: np.array(1.0 / size), owner="all"))

    def gradient(self, output_grad: Node) -> List[Operator]:
        return []


class Reshape(Operator):
    def __init__(self, player: Player, input_nodes: List[Node],
                 original_shape: Tuple[int, ...], new_shape: Tuple[int, ...], name: str = None):
        super(Reshape, self).__init__(player, input_nodes, name or f"Reshape: {original_shape} -> {new_shape}")
        self.original_shape = original_shape
        self.new_shape = new_shape

    def forward(self):
        self.set(self.player.reshape(self.input_nodes[0].get(), self.new_shape))

    def gradient(self, output_grad: Node) -> List[Operator]:
        return [Reshape(self.player, [output_grad], self.new_shape, self.original_shape)]
