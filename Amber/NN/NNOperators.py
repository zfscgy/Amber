from Amber.NN.ComposedOperator import *


class Dense(ComposedOperator):
    @check_op_input(n_innodes=3)
    def __init__(self, player: Player, input_nodes: List[Node], in_dim: int, out_dim: int, name: str="Dense"):
        x, w, b = input_nodes
        xw_op = Matmul(player, [input_nodes[0], w])
        b_broadcast_op = Broadcast(player, [b], [], xw_op, 1)
        xw_plus_b_op = Add(player, [xw_op, b_broadcast_op])
        self.in_dim = in_dim
        self.out_dim = out_dim
        super(Dense, self).__init__(player, [input_nodes[0], w, b], [xw_op, b_broadcast_op, xw_plus_b_op], name)



