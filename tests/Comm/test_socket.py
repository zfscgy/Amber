import time
import numpy as np
from Amber.Core.Comm.Peer import Peer
from Amber.Core.Utils import parallel, task_async


def test_throughput():
    sender = Peer("127.0.0.1:18000", {"127.0.0.1:18001": "receiver"})
    receiver = Peer("127.0.0.1:18001", {"127.0.0.1:18000": "sender"})
    time.sleep(1)
    parallel([sender.connect_all, receiver.connect_all])
    receiver.reset_counter()

    def send_data():
        for i in range(100):
            data = np.random.normal(0, 1, [1000, 1000])
            sender.send("receiver", "data", data)

    def recv_data():
        for i in range(100):
            receiver.recv("sender", "data")

    start_time = time.time()
    parallel([send_data, recv_data])
    time_elapsed = time.time() - start_time
    print(f"Time elapsed {time_elapsed:.2f}")
    transported_mb = receiver.traffic_counter_from['sender'] / 1024
    print(f"Data sent {transported_mb} Mbytes")
    print(f"Network speed: {transported_mb / time_elapsed} Mbps")


def test_multiplex():
    sender = Peer("127.0.0.1:18000", {"127.0.0.1:18001": "receiver"})
    receiver = Peer("127.0.0.1:18001", {"127.0.0.1:18000": "sender"})
    time.sleep(1)
    parallel([sender.connect_all, receiver.connect_all])
    receiver.reset_counter()
    time.sleep(1)
    def send_data():
        for i in range(5):
            data = np.random.normal(0, 1, [1000, 1000])
            task_async(lambda: (sender.send("receiver", str(i), data)))

    def recv_data():
        for i in range(5):
            task_async(lambda i=i: (receiver.recv("sender", str(4 - i)), print(f"Finished receiving {5 - i}")))

    parallel([send_data, recv_data])


if __name__ == '__main__':
    test_multiplex()
