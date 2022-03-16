from Amber.Core.Player import ASPlayer
from Amber.NN.ComposedOperator import *
from Amber.NN import MSEError
from Amber.Core.Utils import parallel
from Amber.Tools.TestUtils import array_close


addr_dict = {
    "player0": "127.0.0.1:9800",
    "player1": "127.0.0.1:8101",
    "third-party": "127.0.0.1:8102"
}
player0 = ASPlayer("player0", addr_dict)
player1 = ASPlayer("player1", addr_dict)
third_party = ASPlayer("third-party", addr_dict)


parallel([player0.init_play, player1.init_play, third_party.init_play])


def test_mean():
    x = np.random.normal(0, 1, [100, 100])
    y = np.random.normal(0, 1, [100, 100])

    def mean(player: ASPlayer):
        tensor_x = player.new_tensor(lambda: x)
        tensor_y = player.new_tensor(lambda: y)
        tensor_xy = player.matmul(tensor_x, tensor_y)
        node_x = Node(tensor_xy)
        mean_op = Mean(player, [node_x], [], n_heading_axes=2)
        mean_op.forward()
        if player in [player0, player1]:
            return player.decode(player.reveal(mean_op.get()))
        else:
            return player.reveal(mean_op.get())

    expected_mean = np.mean(x @ y)

    outs, errs = parallel(mean, [(player0,), (player1,), (third_party,)])
    assert errs[0] is None and errs[1] is None and errs[2] is None
    assert array_close(outs[0], expected_mean) and array_close(outs[1], expected_mean) and outs[2] is None, \
        f"Test mean failed, max err {np.max(np.abs(outs[0] - expected_mean))}"


def test_square():
    x = np.random.normal(0, 1, [100])
    y = np.random.normal(0, 1, [100])

    def square(player: ASPlayer):
        tensor_x = player.new_tensor(lambda: x)
        tensor_y = player.new_tensor(lambda: y)
        tensor_xy = player.mul(tensor_x, tensor_y)
        node_x = Node(tensor_xy)
        square_op = Square(player, [node_x], [], n_heading_axes=2)
        square_op.forward()
        if player in [player0, player1]:
            return player.decode(player.reveal(square_op.get()))
        else:
            return player.reveal(square_op.get())

    expected_square = np.square(x * y)

    outs, errs = parallel(square, [(player0,), (player1,), (third_party,)])

    assert errs[0] is None and errs[1] is None and errs[2] is None
    assert array_close(outs[0], expected_square) and array_close(outs[1], expected_square) and outs[2] is None, \
        f"Test mean failed, max err {np.max(np.abs(outs[0] - expected_square))}"


def test_mse():
    x = np.random.normal(0, 10, [100])
    y = np.random.normal(0, 10, [100])

    def mse(player: ASPlayer):
        tensor_x = player.new_tensor(lambda: x, owner="player0")
        tensor_y = player.new_tensor(lambda: y, owner="player1")
        tensor_xy = player.mul(tensor_x, tensor_y)
        node_x = Node(tensor_x)
        node_xy = Node(tensor_xy)
        mse_op = MSEError(player, [node_x, node_xy], 1)
        mse_op.forward()
        mse_grad_op0 = mse_op.gradient(Node(player.new_tensor(lambda: 0.1, owner="player0")))[0]
        mse_grad_op0.forward()

        if player in [player0, player1]:
            return player.decode(player.reveal(mse_op.get())), \
                   player.decode(player.reveal(mse_grad_op0.get()))
        else:
            return player.reveal(mse_op.get()), player.reveal(mse_grad_op0.get())

    expected_mse = np.mean(np.square(x - x * y))
    expected_mse_grad0 = 1 / 100 * 2 * (x - x * y) * 0.1

    outs, errs = parallel(mse, [(player0,), (player1,), (third_party,)])

    assert errs[0] is None and errs[1] is None and errs[2] is None

    assert array_close(outs[0][0], expected_mse) and array_close(outs[1][0], expected_mse) and outs[2][0] is None, \
        f"Test mse failed, max err {np.max(np.abs(outs[0][0] - expected_mse))}"

    assert array_close(outs[0][1], expected_mse_grad0) and array_close(outs[1][1], expected_mse_grad0) \
           and outs[2][1] is None, \
           f"Test mse failed, max err {np.max(np.abs(outs[0][0] - expected_mse_grad0))}"


if __name__ == '__main__':
    for i in range(10000):
        print(f"Test mse {i}")
        test_mse()
