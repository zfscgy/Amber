import time
import pandas as pd
from Amber.Core.Player import LocalPlayer
from Amber.NN import *
from Amber.Tools import NPDataLoader


player = LocalPlayer()
opf = OpFactory(player)
opf.use_as_default()

record_df = pd.DataFrame(columns=['n_batches', 'time', 'acc'])

"""
MNIST Example
"""

mnist_data = pd.read_csv("../Data/Datasets/mnist.csv", header=None).values.astype(np.float)
mnist_data[:, :784] = mnist_data[:, :784] / 10 - 0.5
data_loader = NPDataLoader(mnist_data[:50000, :])
test_xs = mnist_data[50000:55000, :784]
test_ys = mnist_data[50000:55000, 784:]

dense1 = DenseLayer(784, 128)
dense2 = DenseLayer(128, 32)
dense3 = DenseLayer(32, 10)


def network(x):
    x1 = opf.relu(dense1(x), 1/6)
    x2 = opf.relu(dense2(x1), 1/6)
    x3 = opf.sigmoid(dense3(x2))
    return x3


start_time = time.time()
for i in range(300001):
    if i % 1000 == 0:
        xs = opf.new_node(test_xs)
        pred_ys = network(xs).get().value
        acc = np.mean(np.argmax(pred_ys, axis=-1) == np.argmax(test_ys, axis=-1))
        print(f"Round {i} Acc {acc:.4f}")
        record_df = record_df.append({
            'n_batches': i,
            'time': time.time() - start_time,
            'acc': acc
        }, ignore_index=True)

    batch = data_loader.get_batch(64)
    xs = opf.new_node(batch[:, :784])
    ys = opf.new_node(batch[:, 784:])
    pred_ys = network(xs)
    loss = opf.mean_square_error(pred_ys, ys, 2)
    print(f"Round {i} train loss: {loss.get().value}")
    paras = dense1.parameters() + dense2.parameters() + dense3.parameters()
    gradients = opf.gradient_on(loss, paras, 0.05)
    for para, grad in zip(paras, gradients):
        para.set((para - grad).get())

    opf.clear_graph()

    if False:
        g0 = new_para_tensors[0].value
        g1 = new_para_tensors[1].value
        g2 = new_para_tensors[2].value
        g3 = new_para_tensors[3].value
        print(np.max(np.abs(g0)), np.max(np.abs(g1)), np.max(np.abs(g2)), np.max(np.abs(g3)))
        print(np.std(g0), np.std(g1), np.std(g2), np.std(g3))


record_df.to_csv("record_dnn_mnist_local_784_128_10.csv")
