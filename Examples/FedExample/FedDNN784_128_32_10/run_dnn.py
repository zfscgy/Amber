import time
import pandas as pd
from Amber.Core.Player import RTASPlayer
from Amber.NN import *
from Amber.Tools import NPDataLoader



addr_dict = {
    "player0": "127.0.0.1:9100",
    "player1": "127.0.0.1:9101",
    "third-party": "127.0.0.1:9102"
}



def run_player(rolename: str):
    record_df = pd.DataFrame(columns=['n_batches', 'time', 'acc'])
    player = RTASPlayer(rolename, addr_dict)
    time.sleep(5)
    player.init_play()
    time.sleep(3)
    opf = OpFactory(player)
    opf.use_as_default()

    """
    MNIST Example
    """
    mnist_data = pd.read_csv("../../Data/Datasets/mnist.csv", header=None).values.astype(np.float)
    print(f"{rolename}: data read finished")
    mnist_data[:, :784] = mnist_data[:, :784] / 10 - 0.5
    data_loader = NPDataLoader(mnist_data[:50000, :])
    test_xs = mnist_data[50000:55000, :784]
    test_ys = mnist_data[50000:55000, 784:]

    count_start_time = 0

    def test_speed_start():
        nonlocal count_start_time
        if player.role == "player0":
            count_start_time = time.time()
            player.peer.reset_counter()

    def test_speed_end():
        if player.role == "player0":
            count_time = time.time() - count_start_time
            print(f"Network speed with player1: "
                  f"from {player.peer.traffic_counter_from['player1'] / count_time / 1024:.2f} Mbps "
                  f"to {player.peer.traffic_counter_to['player1'] / count_time / 1024:.2f} Mbps ")
            print(f"Network speed with third-party: "
                  f"from {player.peer.traffic_counter_from['third-party'] / count_time / 1024:.2f} Mbps "
                  f"to {player.peer.traffic_counter_to['third-party'] / count_time / 1024:.2f} Mbps ")


    dense1 = DenseLayer(784, 128)
    dense2 = DenseLayer(128, 32)
    dense3 = DenseLayer(32, 10)

    def network(x):
        x1 = opf.relu(dense1(x), 1 / 6)
        x2 = opf.relu(dense2(x1), 1 / 6)
        x3 = opf.sigmoid(dense3(x2))
        return x3

    start_time = time.time()
    for i in range(300001):
        if i % 1000 == 0:
            player.disable_triple_buffer = True
            xs = opf.new_node(test_xs)
            ys = network(xs)
            if player.role in ["player0", "player1"]:
                pred_ys = player.decode(player.reveal(ys.get()))
            if player.role == "player0":
                acc = np.mean(np.argmax(pred_ys, axis=-1) == np.argmax(test_ys, axis=-1))
                print(f"Round {i} Acc {acc:.4f}")
                record_df = record_df.append({
                    'n_batches': i,
                    'time': time.time() - start_time,
                    'acc': acc
                }, ignore_index=True)
                record_df.to_csv("record_dnn_mnist_fed_784-128-32-10.csv")
            player.disable_triple_buffer = False

        batch = data_loader.get_batch(64)
        xs = opf.new_node(lambda: batch[:, :784])
        ys = opf.new_node(lambda: batch[:, 784:])
        pred_ys = network(xs)
        loss = opf.mean_square_error(pred_ys, ys, 2)
        if player.role in ["player0", "player1"]:
            loss_val = player.decode(player.reveal(loss.get()))
            if player.role == "player0":
                print(f"Round {i} train loss: {loss_val:.3f}")

        paras = dense1.parameters() + dense2.parameters() + dense3.parameters()
        gradients = opf.gradient_on(loss, paras, 0.05)
        for para, grad in zip(paras, gradients):
            para.set((para - grad).get())

        opf.clear_graph()
    if player.role == "player0":
        record_df.to_csv("record_dnn_mnist_fed_784-128-32-10.csv")
