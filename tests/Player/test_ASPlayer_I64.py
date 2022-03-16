import numpy as np
from Amber.Tools.TestUtils import array_close
from Amber.Core.Player import ASPlayer
from Amber.Core.Utils import parallel


"""
Test ASPlayer with I64 Backend
"""

addr_dict = {
    "player0": "127.0.0.1:9100",
    "player1": "127.0.0.1:9101",
    "third-party": "127.0.0.1:9102"
}
player0 = ASPlayer("player0", addr_dict)
player1 = ASPlayer("player1", addr_dict)
third_party = ASPlayer("third-party", addr_dict)


def test_init_player():
    outs, errs = parallel([player0.init_play, player1.init_play, third_party.init_play])
    assert errs[0] is None and errs[1] is None and errs[2] is None


def test_new_tensor():
    def new_value_p0():
        return 99

    outs, errs = parallel([player0.new_tensor, player1.new_tensor, third_party.new_tensor],
                          [(new_value_p0, "player0"), (new_value_p0, "player0"), (new_value_p0, "player0")])

    assert errs[0] is None and errs[1] is None and errs[2] is None
    assert outs[0].value == np.array(99 * 2 ** 24) and outs[1].value == np.array(0) and outs[2].value == np.array(0)


def test_reveal():
    def new_value_p0():
        return 99

    outs, _ = parallel([player0.new_tensor, player1.new_tensor, third_party.new_tensor],
                          [(new_value_p0, "player0"), (new_value_p0, "player0"), (new_value_p0, "player0")])

    outs, errs = parallel([player0.reveal, player1.reveal, third_party.reveal], [(out, ) for out in outs])
    assert errs[0] is None and errs[1] is None and errs[2] is None
    assert outs[0].value == np.array(99 * 2 ** 24) and outs[1].value == np.array(99 * 2 ** 24) and outs[2] is None


def test_add():
    def get_mat_x():
        return np.array([[1, 2], [3, 4]])

    def get_mat_y():
        return np.array([[5, 6], [7, 8]])

    def add(player: ASPlayer):
        x = player.new_tensor(get_mat_x, "all")
        y = player.new_tensor(get_mat_y, "player1")
        prod = player.sub(x, y)
        prod_plain_text = player.reveal(prod)
        if player.is_third_party:
            return prod_plain_text
        else:
            return player.decode(prod_plain_text)

    desired_out = np.subtract([[1, 2], [3, 4]], [[5, 6], [7, 8]])

    outs, errs = parallel(add, [(player0,), (player1,), (third_party,)])
    print(f"Expected: {desired_out}, get {outs[0]}")
    assert errs[0] is None and errs[1] is None and errs[2] is None
    assert array_close(outs[0], desired_out) and array_close(outs[1], desired_out)


def test_mul():
    x = np.random.normal(0, 10, [100])
    y = np.random.normal(0, 10, [100])
    expected_xy = np.multiply(x, y)

    def mul(player: ASPlayer):
        tensor_x = player.new_tensor(lambda: x)
        tensor_y = player.new_tensor(lambda: y)
        xy = player.mul(tensor_x, tensor_y)
        xy_reveal = player.reveal(xy)
        if player in [player0, player1]:
            xy_reveal = player.decode(xy_reveal)
        return xy_reveal

    outs, errs = parallel(mul, [(player0,), (player1,), (third_party,)])

    assert errs[0] is None and errs[1] is None and errs[2] is None, \
        f"Error: {errs}"
    max_err = np.max(np.abs(expected_xy - outs[0]))
    assert array_close(outs[0], expected_xy) and array_close(outs[1], expected_xy) and outs[2] is None, \
        f"Calculation failed, max error: {max_err}"


def test_mul_local():
    x = np.random.normal(0, 10, [1, 100])
    y = np.random.normal(0, 1, [100, 1])
    z = np.random.normal(0, 10, [1, 1])
    expected_xyz = (x @ y) * z

    def mul_local(player: ASPlayer):
        tensor_x = player.new_tensor(lambda: x, owner="player0")
        tensor_y = player.new_tensor(lambda: y, owner="player0")
        tensor_z = player.new_tensor(lambda: z, owner="all")
        tensor_xy = player.matmul(tensor_x, tensor_y)
        tensor_xyz = player.mul(tensor_xy, tensor_z)
        xyz_reveal = player.reveal(tensor_xyz)
        if player in [player0, player1]:
            xyz_reveal = player.decode(xyz_reveal)
        return xyz_reveal

    outs, errs = parallel(mul_local, [(player0,), (player1,), (third_party,)])

    assert errs[0] is None and errs[1] is None and errs[2] is None, \
        f"Error: {errs}"
    max_err = np.max(np.abs(expected_xyz - outs[0]))
    assert array_close(outs[0], expected_xyz) and array_close(outs[1], expected_xyz) and outs[2] is None, \
        f"Calculation failed, max error: {max_err}"


def test_matmul():
    x = np.random.normal(0, 10, [np.random.randint(50, 52), 100])
    y = np.random.normal(0, 10, [100, np.random.randint(150, 152)])
    expected_xy = np.matmul(x, y)

    shares = dict()

    def matmul(player: ASPlayer):
        nonlocal shares
        player.disable_triple_buffer = True
        tensor_x = player.new_tensor(lambda: x)
        tensor_y = player.new_tensor(lambda: y)
        xy = player.matmul(tensor_x, tensor_y)
        shares[player] = xy
        xy_reveal = player.reveal(xy)
        if player in [player0, player1]:
            xy_reveal = player.decode(xy_reveal)
        return xy_reveal

    outs, errs = parallel(matmul, [(player0,), (player1,), (third_party,)])

    assert errs[0] is None and errs[1] is None and errs[2] is None, \
        f"Error: {errs}"
    max_err = np.max(np.abs(expected_xy - outs[0]))
    assert array_close(outs[0], expected_xy) and array_close(outs[1], expected_xy) and outs[2] is None, \
        f"Calculation failed, max error: {max_err}"


def test_matmul_local():
    x = np.random.normal(0, 5, [10, 10])
    w = np.random.normal(0, 5, [10, 10])
    y = np.random.normal(0, 5, [10, 5])
    expected_xy = x @ w @ y

    def matmul(player: ASPlayer):
        tensor_x = player.new_tensor(lambda: x, "player0")
        tensor_w = player.new_tensor(lambda: w, "player1")
        tensor_y = player.new_tensor(lambda: y, "all")
        xw = player.matmul(tensor_x, tensor_w)
        xwy = player.matmul(xw, tensor_y)
        xy_reveal = player.reveal(xwy)
        if player in [player0, player1]:
            xy_reveal = player.decode(xy_reveal)
        return xy_reveal

    outs, errs = parallel(matmul, [(player0,), (player1,), (third_party,)])

    assert errs[0] is None and errs[1] is None and errs[2] is None, \
        f"Error at round: {errs}"
    print(f"Expected: {expected_xy}, get {outs[0]}")
    max_err = np.max(np.abs(expected_xy - outs[0]))
    assert array_close(outs[0], expected_xy) and array_close(outs[1], expected_xy) and outs[2] is None, \
        f"Calculation failed at round, max error: {max_err}"



def test_sum():
    x = np.random.normal(0, 5, [100, 100])
    y = np.random.normal(0, 5, [100, 100])

    def sum(player: ASPlayer):
        x_tensor = player.new_tensor(lambda: x)
        y_tensor = player.new_tensor(lambda: y)
        xy = player.matmul(x_tensor, y_tensor)
        sum_xy = player.reveal(player.sum(xy))
        if player in [player0, player1]:
            sum_xy = player.decode(sum_xy)
        return sum_xy

    expected_sum_xy = np.sum(x @ y)
    outs, errs = parallel(sum, [(player0,), (player1,), (third_party,)])

    assert errs[0] is None and errs[1] is None and errs[2] is None
    assert array_close(expected_sum_xy, outs[0]) and array_close(expected_sum_xy, outs[1]) and outs[2] is None, \
        f"Test sum failed, max error {np.max(np.abs(outs[0] - expected_sum_xy))}"


def test_sum_many_times():
    for i in range(1000):
        print(f"Test sum many times: {i}")
        test_sum()


if __name__ == '__main__':
    test_init_player()
    for i in range(1000):
        print(i)
        test_add()