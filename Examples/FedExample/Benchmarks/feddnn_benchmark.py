import time
import numpy as np
from Amber.Core.Player import RTASPlayer
from Amber.NN import OpFactory, DenseLayer
from Amber.Core.Utils import parallel, parallel_process



addr_dict = {
    "player0": "127.0.0.1:9200",
    "player1": "127.0.0.1:9201",
    "third-party": "127.0.0.1:9202"
}
player0 = RTASPlayer("player0", addr_dict)
player1 = RTASPlayer("player1", addr_dict)
third_party = RTASPlayer("third-party", addr_dict)


def reset_all_counter():
    player0.peer.reset_counter()
    player1.peer.reset_counter()
    third_party.peer.reset_counter()



def init_all():
    parallel([player0.init_play, player1.init_play, third_party.init_play])
    print("All party initialized")


def benchmark_lr(dim: int, batch_size: int, out_dim: int=1):
    print("=======benchmark lr============")
    def benchmark_player(player: RTASPlayer):
        player.triple_buffer_size = 1
        opf = OpFactory(player)
        # opf.use_as_default()  # In multithread, cannot use this since variables are shared!!!
        dense1 = DenseLayer(dim, out_dim, op_factory=opf)  # For multithreading, the use_as_default method will let all threads' opf with a particular player
        paras = dense1.parameters()
        def net(xs):
            x1 = opf.sigmoid(dense1(xs))
            return x1

        def backprop(xs, ys):
            pred_ys = net(xs)
            '''
            loss = opf.mean_square_error(ys, pred_ys, dim=2)
            grads = opf.gradient_on(loss, paras, 0.05)
            '''

            grad_ys = opf.mul(opf.sub(pred_ys, ys), opf.new_node(0.1 / batch_size, owner='all'))
            # grad_zs = opf.sub(opf.new_node(1, owner='all'), pred_ys)
            grad_zs = opf.mul(opf.mul(grad_ys, pred_ys), opf.sub(opf.new_node(1, owner='all'), pred_ys))
            grad_bias = opf.sum(grad_zs, axis=[0], n_heading_axes=1)
            grad_ws = opf.matmul(opf.transpose(xs), grad_zs)
            grads = [grad_ws, grad_bias]

            for para, grad in zip(paras, grads):
                para.set(opf.sub(para, grad).get())

        xs = opf.new_node(np.random.normal(0, 1, [batch_size, dim]))
        ys = opf.new_node(np.random.normal(0, 1, [batch_size, out_dim]))

        if player.role == "player0":
            reset_all_counter()
            time.sleep(1)
            print("Start benchmarking for inference....")
            start_time = time.time()
        for i in range(10):
            net(xs)

        if player.role == "player0":
            comms = player0.peer.traffic_counter_from['third-party'] + player0.peer.traffic_counter_from['player1'] +\
                    player1.peer.traffic_counter_from['third-party'] + player1.peer.traffic_counter_from['player0'] + \
                    third_party.peer.traffic_counter_from['player0'] + third_party.peer.traffic_counter_from['player1']


            comms_p2 = third_party.peer.traffic_counter_from['player0'] + third_party.peer.traffic_counter_from['player1'] + \
                       third_party.peer.traffic_counter_to['player0'] + third_party.peer.traffic_counter_to['player1']

            time_elapsed = time.time() - start_time

            print(f"Training Time per iteration {time_elapsed / 10:.3f}s, "
                  f"communication {comms / (1024**2 * 10):.3f}Mb, "
                  f"p2: {comms_p2 / (1024**2 * 10):.3f}Mb")


        if player.role == "player0":
            reset_all_counter()
            time.sleep(1)
            print("Start benchmarking for training one batch....")
            start_time = time.time()
        for i in range(10):
            backprop(xs, ys)

        if player.role == "player0":
            comms = player0.peer.traffic_counter_from['third-party'] + player0.peer.traffic_counter_from['player1'] +\
                    player1.peer.traffic_counter_from['third-party'] + player1.peer.traffic_counter_from['player0'] + \
                    third_party.peer.traffic_counter_from['player0'] + third_party.peer.traffic_counter_from['player1']

            comms_p2 = third_party.peer.traffic_counter_from['player0'] + third_party.peer.traffic_counter_from['player1'] + \
                       third_party.peer.traffic_counter_to['player0'] + third_party.peer.traffic_counter_to['player1']

            time_elapsed = time.time() - start_time
            print(f"Training Time per iteration {time_elapsed / 10:.3f}s, "
                  f"communication {comms / (1024**2 * 10):.3f}Mb, "
                  f"p2: {comms_p2 / (1024 ** 2 * 10):.3f}Mb")

    """
    ************
    If to measure traffic, must use parallel!!!
    Because when using parallel_process, the traffic can not be recorded since the player is re-created in the new process!!!
    """
    outs, errs = parallel(benchmark_player, [(player0,), (player1,), (third_party,)])

    print(errs)



def benchmark_dnn(input_dim: int, hidden_dim: int, batch_size: int):
    print("=======benchmark dnn============")
    def benchmark_player(player: RTASPlayer):
        player.triple_buffer_size = 10
        opf = OpFactory(player)
        opf.use_as_default()
        dense1 = DenseLayer(input_dim, hidden_dim, op_factory=opf)  # For multithreading, the use_as_default method will let all threads' opf with a particular player
        dense2 = DenseLayer(hidden_dim, 1, op_factory=opf)
        paras = dense1.parameters() + dense2.parameters()
        def net(xs):
            h1 = opf.relu(dense1(xs), 1/6)
            y = opf.sigmoid(dense2(h1))
            return h1, y

        def backprop(xs, ys):
            h1, pred_ys = net(xs)
            '''
            loss = opf.mean_square_error(ys, pred_ys, dim=2)
            grads = opf.gradient_on(loss, paras, 0.05)
            '''

            grad_ys = opf.mul(opf.sub(pred_ys, ys), opf.new_node(0.05 / batch_size, owner='all'))
            grad_zs = opf.mul(opf.mul(grad_ys, pred_ys), opf.sub(opf.new_node(1, owner='all'), pred_ys))

            grad_bias2 = opf.sum(grad_zs, axis=[0], n_heading_axes=1)
            grad_ws2 = opf.matmul(opf.transpose(h1), grad_zs)

            grad_y1s = opf.matmul(grad_zs, opf.transpose(dense2.w))
            grad_z1s = opf.mul(grad_y1s, opf.relu_grad(grad_y1s, 1/6))

            grad_bias1 = opf.sum(grad_z1s, axis=[0], n_heading_axes=1)
            grad_ws1 = opf.matmul(opf.transpose(xs), grad_z1s)

            grads = [grad_ws1, grad_bias1, grad_ws2, grad_bias2]

            for para, grad in zip(paras, grads):
                para.set(opf.sub(para, grad).get())

        xs = opf.new_node(np.random.normal(0, 1, [batch_size, input_dim]))
        ys = opf.new_node(np.random.normal(0, 1, [batch_size, 1]))

        if player.role == "player0":
            print("Start benchmarking for inference....")
            reset_all_counter()
            time.sleep(1)
            start_time = time.time()

        for i in range(10):
            net(xs)

        if player.role == "player0":
            time_elapsed = time.time() - start_time
            time.sleep(1)
            comms = player0.peer.traffic_counter_from['third-party'] + player0.peer.traffic_counter_from['player1'] +\
                    player1.peer.traffic_counter_from['third-party'] + player1.peer.traffic_counter_from['player0'] + \
                    third_party.peer.traffic_counter_from['player0'] + third_party.peer.traffic_counter_from['player1']
            print(f"Inference Time per iteration {time_elapsed / 10:.3f}s, communication {comms / (1024**2 * 10):.2f}Mb")

        if player.role == "player0":
            print("Start benchmarking for training one batch....")
            reset_all_counter()
            time.sleep(1)
            start_time = time.time()

        for i in range(10):
            backprop(xs, ys)

        if player.role == "player0":
            time_elapsed = time.time() - start_time
            time.sleep(1)
            comms = player0.peer.traffic_counter_from['third-party'] + player0.peer.traffic_counter_from['player1'] +\
                    player1.peer.traffic_counter_from['third-party'] + player1.peer.traffic_counter_from['player0'] + \
                    third_party.peer.traffic_counter_from['player0'] + third_party.peer.traffic_counter_from['player1']
            print(f"Training Time per iteration {time_elapsed / 10:.3f}s, communication {comms / (1024**2 * 10):.2f}Mb")

    outs, errs = parallel(benchmark_player, [(player0,), (player1,), (third_party,)])
    print(errs)


if __name__ == '__main__':
    init_all()
    # benchmark_lr(100, 64)
    # benchmark_lr(100, 128)
    # benchmark_lr(1000, 64)
    # benchmark_lr(1000, 128)
    # benchmark_dnn(1000, 500, 64)
    # benchmark_dnn(1000, 500, 128)

    for k in range(11):
        print(k)
        time.sleep(1)
        benchmark_lr(1000, 64, out_dim=2**k)
