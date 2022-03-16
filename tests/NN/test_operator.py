import pandas as pd

from Amber.Core.Player import LocalPlayer
from Amber.NN import *


player = LocalPlayer()


def test_add_gradient():
    input_node_0 = Node()
    input_node_1 = Node()
    add_op = Add(player, [input_node_0, input_node_1])
    grad_on_sum = Node(player.new_tensor(lambda: 1))
    gradients = add_op.gradient(grad_on_sum)
    gradients[0].forward()
    gradients[1].forward()

    assert gradients[0].get().value == 1
    assert gradients[1].get().value == 1


def test_mul_gradient():
    input_node_0 = Node(player.new_tensor(lambda: 1))
    input_node_1 = Node(player.new_tensor(lambda: 2))
    mul_op = Mul(player, [input_node_0, input_node_1])
    grad_on_prod = Node(player.new_tensor(lambda: 3))
    gradients = mul_op.gradient(grad_on_prod)
    gradients[0].forward()
    gradients[1].forward()

    assert gradients[0].get().value == 6
    assert gradients[1].get().value == 3


def test_broadcast_gradient():
    x = Node(player.new_tensor(lambda: np.array([1])))
    y = Node(player.new_tensor(lambda: np.zeros([5, 2, 3])))
    xx = Broadcast(player, [x], [0], y, 2)
    xx.forward()
    assert np.allclose(xx.get().value, np.ones([5, 2, 3]))
    grad_on_xx = Node(player.new_tensor(lambda: np.ones([5, 2, 3])))
    grad_x = xx.gradient(grad_on_xx)[0]
    grad_x.forward()
    assert grad_x.get().value == 30


def test_sum_gradient():
    x = Node(player.new_tensor(lambda: np.array([[1, 2, 3], [4, 5, 6]])))
    sum_x = Sum(player, [x], [0, -1], 2)
    sum_x.forward()
    assert np.allclose(sum_x.get().value, np.array(21))
    grad_on_sum = Node(player.new_tensor(lambda: np.array(1)))
    grad_x = sum_x.gradient(grad_on_sum)[0]
    grad_x.forward()
    assert np.allclose(grad_x.get().value, np.ones([2, 3]))


def test_composed_gradient():
    x = Node(player.new_tensor(lambda: 2))
    y = Node(player.new_tensor(lambda: 3))
    x_plus_y = Add(player, [x, y])
    x_mul__x_plus_y = Mul(player, [x_plus_y, x])
    composed_op = ComposedOperator(player, [x, y], [x_plus_y, x_mul__x_plus_y])
    composed_op.forward()
    assert composed_op.get().value == 10

    grads_on_output = Node(player.new_tensor(lambda: 1))
    gradient_x, gradient_y = composed_op.gradient(grads_on_output)
    gradient_x.forward()
    gradient_y.forward()
    assert gradient_x.get().value == 7
    assert gradient_y.get().value == 2


def test_matmul_gradient():
    x = Node(player.new_tensor(lambda: np.array([[1, 2, 3, 4]])))
    y = Node(player.new_tensor(lambda: np.array([[4], [3], [2], [1]])))
    x_matmul_y = Matmul(player, [x, y])
    x_matmul_y.forward()
    assert x_matmul_y.get().value[0, 0] == 20

    grads_on_output = Node(player.new_tensor(lambda: np.array([[1]])))
    grad_x, grad_y = x_matmul_y.gradient(grads_on_output)
    grad_x.forward()
    grad_y.forward()

    assert np.allclose(grad_x.get().value, np.array([[4, 3, 2, 1]]))
    assert np.allclose(grad_y.get().value, np.array([[1], [2], [3], [4]]))


def test_sigmoid_gradient():
    x = Node(player.new_tensor(lambda: 1))
    sigmoid = Sigmoid(player, [x])
    sigmoid.forward()
    actual_sigmoid = 1 / (1 + np.exp(-1))
    assert sigmoid.get().value == actual_sigmoid
    sigmoid_grad = sigmoid.gradient(Node(player.new_tensor(lambda: 1)))[0]
    sigmoid_grad.forward()
    assert sigmoid_grad.get().value == actual_sigmoid * (1 - actual_sigmoid)


def test_tanh_gradient():
    x = Node(player.new_tensor(lambda: 2))
    tanh = Tanh(player, [x])
    tanh.forward()
    assert np.allclose(tanh.get().value, np.tanh(2))
    tanh_grad = tanh.gradient(Node(player.new_tensor(lambda: 1)))[0]
    tanh_grad.forward()
    assert np.allclose(tanh_grad.get().value, 1 - np.tanh(2) ** 2)


def test_mse_gradient():
    x = np.array([1, 2, 3, 4])
    y = np.array([4, 3, 2, 1])
    node_x = Node(player.new_tensor(lambda: x))
    node_y = Node(player.new_tensor(lambda: y))
    mse_op = MSEError(player, [node_x, node_y], 1)
    mse_op.forward()
    assert np.allclose(mse_op.get().value, np.mean(np.square(x - y)))

    mse_grad0_op = mse_op.gradient(Node(player.new_tensor(lambda: 0.1)))[0]
    mse_grad0_op.forward()

    assert np.allclose(mse_grad0_op.get().value, 0.25 * 2 * (x - y) * 0.1)


if __name__ == '__main__':
    test_mse_gradient()
