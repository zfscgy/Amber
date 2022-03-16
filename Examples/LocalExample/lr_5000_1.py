import time
import pandas as pd
from Amber.Core.Player import LocalPlayer
from Amber.NN import *
from Amber.Tools import NPDataLoader
from Examples.Data.DataReader import get_gisette_train_test


player = LocalPlayer()
opf = OpFactory(player)
opf.use_as_default()

record_df = pd.DataFrame(columns=['n_batches', 'time', 'acc'])


train_xs, train_ys, test_xs, test_ys = get_gisette_train_test()
train_ys = train_ys[:, np.newaxis]
test_ys = test_ys[:, np.newaxis]
train_loader = NPDataLoader(np.hstack([train_xs, train_ys]))


dense_layer = DenseLayer(5000, 1)


def logistic_regression(xs: Node):
    return opf.tanh(dense_layer(xs))

start_time = time.time()
for i in range(10000):
    if i % 100 == 0:
        pred_test_ys = logistic_regression(opf.new_node(test_xs)).get().value
        acc = np.mean((pred_test_ys > 0).astype(np.float) * 2 - 1 == test_ys)
        print(f"Round {i}, acc {acc:.3f}")
        record_df = record_df.append({"n_batches": i, "time": time.time() - start_time, "acc": acc}, ignore_index=True)

    train_batch = train_loader.get_batch(64)
    train_xs = train_batch[:, :-1]
    train_ys = train_batch[:, -1:]
    pred_ys = logistic_regression(opf.new_node(train_xs))
    loss = opf.mean_square_error(opf.new_node(train_ys), pred_ys, dim=2)
    # print(loss.get().value)
    grads = opf.gradient_on(loss, dense_layer.parameters(), 0.002)
    for para, grad in zip(dense_layer.parameters(), grads):
        para.set((para - grad).get())
    opf.clear_graph()

record_df.to_csv("record_lr_gisette_local_5000_1.csv")
