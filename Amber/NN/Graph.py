from typing import List
from Amber.NN.Operator import *
from Amber.NN.ComposedOperator import *


class GraphException:
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self):
        return self.msg


class Graph(ComposedOperator):
    def __init__(self, player: Player, inputs: List[Node], parameters: List[Node], operators: List[Operator]):
        super(Graph, self).__init__(player, inputs + parameters, operators)
        self.inputs = inputs
        self.parameters = parameters
        self.fixed_paras = set()

        self.output_grad_node = Node()
        self.all_gradient_ops = self.gradient(self.output_grad_node)
        self.input_gradient_nodes = dict()
        self.parameter_gradient_nodes = dict()
        for input_tensor, gradient in zip(self.input_nodes, self.all_gradient_ops):
            if input_tensor in self.inputs:
                self.input_gradient_nodes[input_tensor] = gradient
            else:
                self.parameter_gradient_nodes[input_tensor] = gradient

    def feed(self, input_tensors: List[Tensor]):
        for input_tensor, input_node in zip(input_tensors, self.inputs):
            input_node.set(input_tensor)

    def clear_cache(self):
        """
        The gradient graphs are contained in the Graph, not explicitly shown, so the cache should be also cleared
        (since we use the Graph.compute_gradients method to compute gradients, not using the gradient operetor,
        :return:
        """
        super(Graph, self).clear_cache()
        for grad in self.all_gradient_ops:
            grad.clear_cache()

    def compute_gradients(self, output_grad_tensor: Tensor, include_input=False):
        self.output_grad_node.set(output_grad_tensor)
        for gradient in self.all_gradient_ops:
            if isinstance(gradient, ComposedOperator):
                gradient.forward(use_cache=True)
            else:
                gradient.forward()
        if include_input:
            return [self.parameter_gradient_nodes[p] for p in self.parameters], \
                   [self.input_gradient_nodes[i] for i in self.inputs]
        else:
            return [self.parameter_gradient_nodes[p] for p in self.parameters]

    def set_para_fixed(self, fixed_paras: List[Node], fixed=True):
        for fixed_para in fixed_paras:
            if fixed_para not in self.parameters:
                raise GraphException(f"{fixed_para} not in the graph's parameters list")
            else:
                if fixed:
                    self.fixed_paras.add(fixed_para)
                else:
                    self.fixed_paras.remove(fixed_para)

    def update_parameters(self, new_para_tensors: List[Tensor]):
        for para, new_para_tensor in zip(self.parameters, new_para_tensors):
            if para not in self.fixed_paras:
                para.set(new_para_tensor)
