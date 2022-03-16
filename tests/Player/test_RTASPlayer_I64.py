import numpy as np
from Amber.Core.Player import RTASPlayer
from Amber.Core.Utils import parallel
from Amber.Tools.TestUtils import array_close


"""
Test ASPlayer with I64 Backend
"""
addr_dict = {
    "player0": "127.0.0.1:9110",
    "player1": "127.0.0.1:9111",
    "third-party": "127.0.0.1:9112"
}
player0 = RTASPlayer("player0", addr_dict)
player1 = RTASPlayer("player1", addr_dict)
third_party = RTASPlayer("third-party", addr_dict)


def test_init_player():
    outs, errs = parallel([player0.init_play, player1.init_play, third_party.init_play])
    assert errs[0] is None and errs[1] is None and errs[2] is None


def test_sigmoid():
    def sigmoid(player: RTASPlayer):
        x = player.new_tensor(lambda: np.array([1, 2, 3]))
        y = player.sigmoid(x)
        y_plain_text = player.reveal(y)
        if player.is_third_party:
            return y_plain_text.value
        else:
            return player.decode(y_plain_text)

    desired_y = 1 / (1 + np.exp(-np.array([1, 2, 3])))
    outs, errs = parallel(sigmoid, [(player0,), (player1,), (third_party,)])
    print(outs)
    print(f"Desired: {desired_y}, get: {outs[0]}, {outs[1]}")
    assert errs[0] is None and errs[1] is None and errs[2] is None
    assert array_close(outs[0], desired_y) and array_close(outs[1], desired_y)


def test_sigmoid_mean():
    raw_x = np.random.normal(0, 100000, [10])

    def sigmoid_mean(player: RTASPlayer):
        x = player.new_tensor(lambda: raw_x)
        y = player.sigmoid(x)
        y = player.mul(player.sum(y), 1/10)
        y_plain_text = player.reveal(y)
        if player.is_third_party:
            return y_plain_text.value
        else:
            return player.decode(y_plain_text)

    desired_y = np.mean(1 / (1 + np.exp(-raw_x)))
    outs, errs = parallel(sigmoid_mean, [(player0,), (player1,), (third_party,)])
    print(outs)
    print(f"Desired: {desired_y}, get: {outs[0]}, {outs[1]}")
    assert errs[0] is None and errs[1] is None and errs[2] is None
    assert array_close(outs[0], desired_y) and array_close(outs[1], desired_y)


if __name__ == '__main__':
    test_init_player()
    for i in range(1000):
        print(i)
        test_sigmoid()
