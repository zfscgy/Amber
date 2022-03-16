import time
import pandas as pd
from Amber.Core.Player import RTASPlayer
from Amber.NN import *
from Amber.Tools import NPDataLoader
from Examples.Data.DataReader import get_gisette_train_test


addr_dict = {
    "player0": "127.0.0.1:9100",
    "player1": "127.0.0.1:9101",
    "third-party": "127.0.0.1:9102"
}


def run_lr(rolename: str):
    player = RTASPlayer(rolename, addr_dict)
    time.sleep(5)
    player.init_play()
    opf = OpFactory(player)
    opf.use_as_default()

    record_df = pd.DataFrame(columns=['n_batches', 'time', 'acc'])

    train_xs, train_ys, test_xs, test_ys = get_gisette_train_test()
    train_ys = train_ys[:, np.newaxis]

    test_xs = test_xs[:5000]
    test_ys = test_ys[:5000, np.newaxis]
    train_loader = NPDataLoader(np.hstack([train_xs, train_ys]))


    dense_layer = DenseLayer(5000, 1)


    def logistic_regression(xs: Node):
        return opf.tanh(dense_layer(xs))


    player.triple_buffer_size = 16
    start_time = time.time()
    for i in range(10000):
        if i % 100 == 0:
            player.disable_triple_buffer = True
            pred_test_ys = player.reveal(logistic_regression(opf.new_node(test_xs)).get()).value
            acc = np.mean((pred_test_ys > 0).astype(np.float) * 2 - 1 == test_ys)
            print(f"Round {i}, acc {acc:.3f}")
            if rolename == "player0":
                record_df = record_df.append({"n_batches": i, "time": time.time() - start_time, "acc": acc}, ignore_index=True)

            player.disable_triple_buffer = False

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

    if rolename == "player0":
        record_df.to_csv("record_lr_gisette_fed_5000_1.csv")
