from Amber.Core.Player import LocalPlayer
from Amber.NN import *


"""
    A simple example for wx + y = target, fixed x, y, target, using gradient descent to find w
"""


player = LocalPlayer()


x = Node(player.new_tensor(lambda: 1))
w = Node(player.new_tensor(lambda: 0))
y = Node(player.new_tensor(lambda: -2))
target = Node(player.new_tensor(lambda: 1))

x_mul_w = Mul(player, [x, w])
x_mul_w_plus_y = Add(player, [x_mul_w, y])
graph_out = Graph(player, [x, y], [w], [x_mul_w, x_mul_w_plus_y])
diff = Sub(player, [graph_out, target])
loss = Square(player, [diff])
graph_loss = Graph(player, [x, y, target], [w], [graph_out, diff, loss])
for i in range(100):
    graph_loss.clear_cache()
    graph_loss.forward()
    print(f"Round {i} Loss {graph_loss.get().value:.4f}")
    w_grads = graph_loss.compute_gradients(player.new_tensor(lambda: 1))[0]
    print(f"Round {i} w_grad {w_grads.value:.4f}")
    w_updates = player.mul(-0.1, w_grads)
    w.set(player.add(w.get(), w_updates))
    print(f"Round {i} w updated {w.get().value:.4f}")
