from typing import ClassVar
from Amber.NN.Operator import *


ref_node_users = [Broadcast, Sum, Size]


class FixedInputOperator(Operator):
    def __init__(self, operator: Operator, fixed_nodes: List[Node],
                 name: str = "FixedInput"):
        player = operator.player
        super(FixedInputOperator, self).__init__(player, [], name)
        self.input_nodes = [input_node for input_node in operator.input_nodes if input_node not in fixed_nodes]
        self.fixed_nodes = fixed_nodes
        self.op = operator

    def forward(self):
        self.op.forward()
        self.set(self.op.get())

    def clear_cache(self):
        super(FixedInputOperator, self).clear_cache()
        self.op.clear_cache()

    def gradient(self, output_grad: Node) -> List[Operator]:
        original_gradients = self.op.gradient(output_grad)
        grads_on_vars = []
        for original_input_node, original_gradient in zip(self.op.input_nodes, original_gradients):
            if original_input_node in self.input_nodes:
                fixed_input_nodes = []
                for original_gradient_input in original_gradient.input_nodes:
                    if original_gradient_input in self.fixed_nodes:
                        fixed_input_nodes.append(original_gradient_input)
                if len(fixed_input_nodes) != 0:
                    grads_on_vars.append(FixedInputOperator(original_gradient, fixed_input_nodes))
                else:
                    grads_on_vars.append(original_gradient)

        return grads_on_vars


class ComposedOperator(Operator):
    def __init__(self, player: Player, input_nodes: List[Node], operators: List[Operator], name: str = None):
        super(ComposedOperator, self).__init__(player, input_nodes, name)
        self.operators = operators
        self.intermediate_nodes = []
        self.op_outputs_dict = dict()
        self.ref_nodes = []
        for op in operators:
            for op_input in op.input_nodes:
                if op_input not in self.input_nodes and op_input not in self.intermediate_nodes:
                    raise OpException(f"ComposedOperator: all op_inputs must be already computed")
            if isinstance(op, RefOperator):
                if op.ref_node not in self.input_nodes and op.ref_node not in self.intermediate_nodes:
                    raise OpException(f"ComposedOperator: all ref_nodes must be already computed")
            self.intermediate_nodes.append(op)
            self.op_outputs_dict[op] = op

    def forward(self, use_cache=True):
        self.set(None)
        for op in self.operators:
            if not use_cache or op.get() is None:
                op.forward()
        self.set(self.operators[-1].get())

    def clear_cache(self):
        self.set(None)
        for op in self.operators:
            op.clear_cache()

    def gradient(self, output_grad: Node) -> List[Operator]:
        """
        The gradient of a composed operator is several operators corresponding to several input_nodes
        And a specific gradient operator will take 

        :param output_grad:
        :return:
        """


        # node -> gradients_op
        node_gradients_dict = dict()
        # gradients_op -> gradients_op, like a reversed dict
        gradients_node_dict = dict()
        grads_on_last_node_inputs = self.operators[-1].gradient(output_grad)
        for i, node in enumerate(self.operators[-1].input_nodes):
            node_gradients_dict[node] = grads_on_last_node_inputs[i]
            gradients_node_dict[grads_on_last_node_inputs[i]] = grads_on_last_node_inputs[i]

        def add_gradient(node: Node, new_gradient: Operator):
            # Since the new_gradient operator may contain some input nodes which are not the
            # graph inputs or the output_grad node, but some other intermediate nodes' gradients,
            # Here we must compose those gradient operators to make all of the gradients' inputs
            # only contain graph inputs and the output_grad node

            # For example, the graph
            # node1 = x + y, node2 = node1 * x
            # Then the grad on node2 = output_grad * node1
            # The grad on node1 = x * grad(node2)
            # The grad on x = grad(node1) + node1
            # Here the grad(node1) is an intermediate node, so we have to calculate it first.
            # So the x's gradient operator should compose grad(node1)
            new_input_nodes = []
            new_grad_ops = []
            if isinstance(new_gradient, RefOperator):
                ref_nodes = [new_gradient.ref_node]
            else:
                ref_nodes = []

            for input_node in new_gradient.input_nodes + ref_nodes:
                if gradients_node_dict.get(input_node) is not None:
                    for sub_input_node in gradients_node_dict.get(input_node).input_nodes:
                        if sub_input_node not in new_input_nodes:
                            new_input_nodes.append(sub_input_node)
                    new_grad_ops.append(gradients_node_dict.get(input_node))
                else:
                    new_input_nodes.append(input_node)

            new_gradient = ComposedOperator(self.player, new_input_nodes, new_grad_ops + [new_gradient])

            if node not in node_gradients_dict:
                node_gradients_dict[node] = new_gradient
            else:
                previous_gradient = node_gradients_dict[node]
                gradient_sum = Add(self.player, [previous_gradient, new_gradient])
                node_gradients_dict[node] = \
                    ComposedOperator(self.player,
                                     list(set(previous_gradient.input_nodes) | set(new_gradient.input_nodes)),
                                     [previous_gradient, new_gradient, gradient_sum])
            gradients_node_dict[node_gradients_dict[node]] = node_gradients_dict[node]

        for op in reversed(self.operators[:-1]):
            gradients = op.gradient(node_gradients_dict[op])
            for i, node in enumerate(op.input_nodes):
                add_gradient(node, gradients[i])

        gradients = []
        for node in self.input_nodes:
            gradients.append(node_gradients_dict[node])
        self.node_gradients_dict = node_gradients_dict
        return gradients


class Matmul(Operator):
    @check_op_input(n_innodes=2)
    def __init__(self, player: Player, input_nodes: List[Node], name: str="Matmul"):
        super(Matmul, self).__init__(player, input_nodes, name)

    def forward(self):
        self.set(self.player.matmul(self.input_nodes[0].get(), self.input_nodes[1].get()))

    def gradient(self, output_grad: Node) -> List[Operator]:
        in0_t = Transpose(self.player, [self.input_nodes[0]])
        in1_t = Transpose(self.player, [self.input_nodes[1]])
        o_matmul_in1_t = Matmul(self.player, [output_grad, in1_t])
        in0_t_matmul_o = Matmul(self.player, [in0_t, output_grad])

        grad_in0 = ComposedOperator(self.player, [output_grad, self.input_nodes[1]], [in1_t, o_matmul_in1_t])
        grad_in1 = ComposedOperator(self.player, [self.input_nodes[0], output_grad], [in0_t, in0_t_matmul_o])
        return [grad_in0, grad_in1]


class Square(Operator):
    @check_op_input(n_innodes=1)
    def __init__(self, player: Player, input_nodes: List[Node], name: str = "Square"):
        super(Square, self).__init__(player, input_nodes, name)

    def forward(self):
        self.set(self.player.square(self.input_nodes[0].get()))

    def gradient(self, output_grad: Node) -> List[Operator]:
        node_2 = Node(self.player.new_tensor(lambda: 2, owner="all"))
        input_mul_2 = FixedInputOperator(Mul(self.player, [node_2, self.input_nodes[0]]), [node_2])
        input_mul_2_mul_output_grad = Mul(self.player, [input_mul_2, output_grad])
        grad = ComposedOperator(self.player,
                                [self.input_nodes[0], output_grad], [input_mul_2, input_mul_2_mul_output_grad],
                                f"grad_{self.name}")
        return [grad]


class Sigmoid(Operator):
    @check_op_input(n_innodes=1)
    def __init__(self, player: Player, input_nodes: List[Node], name: str = "Sigmoid"):
        super(Sigmoid, self).__init__(player, input_nodes, name)

    def forward(self):
        self.set(self.player.sigmoid(self.input_nodes[0].get()))

    def gradient(self, output_grad: Node) -> List[Operator]:
        one = Node(self.player.new_tensor(lambda: 1, owner='all'))
        one_minus_out_op = FixedInputOperator(Sub(self.player, [one, self], f"{self.name}_1-out"), [one])

        out_mul_one_minus_out_op = Mul(self.player, [self, one_minus_out_op])
        grad_out_op = Mul(self.player, [output_grad, out_mul_one_minus_out_op])

        grad_op = ComposedOperator(self.player, [self, output_grad],
                                   [one_minus_out_op, out_mul_one_minus_out_op, grad_out_op], f"grad_{self.name}")

        return [grad_op]


class Tanh(Operator):
    @check_op_input(n_innodes=1)
    def __init__(self, player: Player, input_nodes: List[Node], name: str = "Sigmoid"):
        super(Tanh, self).__init__(player, input_nodes, name)

    def forward(self):
        self.set(self.player.tanh(self.input_nodes[0].get()))

    def gradient(self, output_grad: Node) -> List[Operator]:
        one = Node(self.player.new_tensor(lambda: 1, owner='all'))
        out_square_op = Square(self.player, [self])
        one_minus_out_square_op = Sub(self.player, [one, out_square_op])
        one_minus_out_square_mul_outgrad_op = Mul(self.player, [output_grad, one_minus_out_square_op])
        grad_op = FixedInputOperator(
            ComposedOperator(self.player, [output_grad, one, self],
                             [out_square_op, one_minus_out_square_op, one_minus_out_square_mul_outgrad_op]),
            [one], f"grad_{self.name}"
        )
        return [grad_op]


class ReluGrad(Operator):
    @check_op_input(n_innodes=1)
    def __init__(self, player: Player, input_nodes: List[Node], k: float):
        super(ReluGrad, self).__init__(player, input_nodes)
        self.k = k

    def forward(self):
        self.set(self.player.relu_grad(self.input_nodes[0].get(), self.k))

    def gradient(self, output_grad: Node) -> List:
        raise NotImplementedError()


class Relu(Operator):
    @check_op_input(n_innodes=1)
    def __init__(self, player: Player, input_nodes: List[Node], k: float):
        super(Relu, self).__init__(player, input_nodes)
        self.k = k

    def forward(self):
        self.set(self.player.relu(self.input_nodes[0].get(), self.k))

    def gradient(self, output_grad: Node) -> List:
        relu_grad = ReluGrad(self.player, [self], self.k)
        mul_relu_grad = Mul(self.player, [relu_grad, output_grad])
        return [ComposedOperator(self.player, [output_grad, self], [relu_grad, mul_relu_grad], f"grad_{self.name}")]


class Mean(ComposedOperator):
    @check_op_input(n_innodes=1)
    def __init__(self, player: Player, input_nodes: List[Node],
                 mean_axes: List[int] = None, n_heading_axes: int = 0,
                 name: str = "Mean"):
        if mean_axes is None:
            mean_axes = []
        self.mean_axes = mean_axes
        self.n_heading_axes = n_heading_axes

        def recalc(axis):
            if axis > 0:
                return n_heading_axes + axis
            else:
                return axis

        summing_axes = list(range(n_heading_axes)) + [recalc(axis) for axis in mean_axes]

        sum_op = Sum(player, input_nodes, summing_axes, n_heading_axes)
        size_op = Size(player, input_nodes[0], summing_axes, reciprocal=True)
        mean_op = Mul(player, [size_op, sum_op])

        super(Mean, self).__init__(player, input_nodes, [sum_op, size_op, mean_op], name)


class MSEError(ComposedOperator):
    @check_op_input(n_innodes=2)
    def __init__(self, player, input_nodes: List[Node], dim: int, name: str="MSEError"):
        diff_op = Sub(player, input_nodes)
        square_op = Square(player, [diff_op])
        mean_op = Mean(player, [square_op], [], dim)
        super(MSEError, self).__init__(player, input_nodes, [diff_op, square_op, mean_op], name)