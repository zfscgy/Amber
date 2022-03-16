import pandas as pd
from Amber.Core.Player import LocalPlayer
from Amber.NN import *
from Amber.Tools import NPDataLoader


player = LocalPlayer()

"""
MNIST Example
"""
mnist_data = pd.read_csv("../Data/Datasets/mnist.csv", header=None).values.astype(np.float)
mnist_data[:, :784] = mnist_data[:, :784] / 20 - 0.5
data_loader = NPDataLoader(mnist_data[:50000, :])
test_xs = mnist_data[50000:55000, :784]
test_ys = mnist_data[50000:55000, 784:]


input_xs = Node()
input_ys = Node()

dense1 = Dense(player, [input_xs], 784, 10)
sigmoid = Sigmoid(player, [dense1])
graph_model = Graph(player, [input_xs], [dense1.w, dense1.b], [dense1, sigmoid])
mse_op = MSEError(player, [graph_model, input_ys], dim=2)
loss_graph = Graph(player, [input_xs, input_ys], [dense1.w, dense1.b], [graph_model, mse_op])


for i in range(10001):
    if i % 100 == 0:
        graph_model.feed([test_xs])
        graph_model.forward()
        pred_ys = graph_model.get().value
        graph_model.clear_cache()
        acc = np.mean(np.argmax(pred_ys, axis=-1) == np.argmax(test_ys, axis=-1))
        print(f"Round {i} Acc {acc:.4f}")

    batch = data_loader.get_batch(32)
    xs = batch[:, :784]
    ys = batch[:, 784:]
    tensor_xs = player.new_tensor(lambda: xs)
    tensor_ys = player.new_tensor(lambda: ys)
    loss_graph.feed([tensor_xs, tensor_ys])
    loss_graph.forward()
    print(f"Current train loss: {loss_graph.get().value}")
    gradients = loss_graph.compute_gradients(player.new_tensor(lambda: 0.02))
    new_para_tensors = [player.sub(para.get(), gradient) for para, gradient in zip(loss_graph.parameters, gradients)]
    loss_graph.update_parameters(new_para_tensors)
    loss_graph.clear_cache()
