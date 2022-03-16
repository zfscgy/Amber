import time
import socket
import threading
from Amber.Core.Utils import parallel
import logging

logger = logging.getLogger("Socket")


len_header = 6


class SocketException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


def read_socket(s: socket.socket) -> bytes:
    try:
        len_bytes = s.recv(len_header)
        if len(len_bytes) == 0:
            raise SocketException("read_socket: no data to read")
        content_len = int.from_bytes(len_bytes, byteorder='big')
        logger.debug("Get message size %d" % content_len)
        content = bytes()
        while len(content) < content_len:
            content += s.recv(content_len - len(content))
        return content
    except:
        raise SocketException("Socket read error")


def write_socket(s: socket.socket, content: bytes):
    try:
        content_len = len(content) + len_header
        len_bytes = len(content).to_bytes(len_header, 'big')
        send_bytes = len_bytes + content
        while content_len != 0:
            content_len -= s.send(send_bytes[-content_len:])

    except:
        raise SocketException("Socket send error")


class SocketServer:
    def __init__(self, address: str, other_addrs: dict, timeout=10):
        """
        :param address:
        :param other_addrs: dict[address, name]
        :param timeout:
        """
        self.addr = address
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            ipv4, port = address.split(":")
            port = int(port)

        except:
            raise SocketException("Address %s not valid" % address)

        # setsockopt should be called before binding the socket.
        # Use SO_REUSEPORT to prevent 'Address already in use' problem
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((ipv4, port))

        logger.debug("Default timeout set to %d" % timeout)
        socket.setdefaulttimeout(timeout)
        self.other_addrs = other_addrs
        self.other_recv_sockets = dict()
        self.other_send_sockets = dict()
        self.send_locks = dict()
        self.listening = True

        self.listen_thread = threading.Thread(target=self._listen_loop)
        self.listen_thread.start()

        # Count traffic from/to
        self.traffic_counter_to = dict()
        self.traffic_counter_from = dict()

    def set_timeout(self,timeout: float):
        logger.debug("Default timeout set to %d" % timeout)
        socket.setdefaulttimeout(timeout)

    def _listen_loop(self):
        self.socket.listen()
        not_connected_others = set(self.other_addrs.keys())
        while self.listening:
            try:
                accpeted_socket, addr = self.socket.accept()
            except TimeoutError as e:
                continue

            try:
                claimed_addr = str(read_socket(accpeted_socket), "utf-8")
            except TimeoutError:
                raise SocketException("Did not receive address claim after connection from %s" % addr)

            if claimed_addr.split(":")[0] != addr[0]:
                raise SocketException("Claimed Address %s do not match with the actual send address %s"
                                      % (claimed_addr, addr[0]))
            if claimed_addr in self.other_addrs:
                self.other_recv_sockets[self.other_addrs[claimed_addr]] = accpeted_socket
                self.traffic_counter_from[self.other_addrs[claimed_addr]] = 0
            else:
                raise SocketException("Get unexpected socket connection from %s" % addr)

            not_connected_others.remove(claimed_addr)
            if len(not_connected_others) == 0:
                break
        self.listening = False

    def connect_all(self):
        def connect_one(peer_addr: str, peer_name: str):
            my_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                peer_ipv4, peer_port = peer_addr.split(":")
                peer_port = int(peer_port)
            except:
                raise SocketException("%s is not a valid address" % peer_addr)

            try:
                my_socket.connect((peer_ipv4, peer_port))
                write_socket(my_socket, self.addr.encode("utf-8"))
            except TimeoutError:
                raise SocketException("Connect to %s: %s failed" % (peer_name, peer_addr))
            self.other_send_sockets[peer_name] = my_socket
            self.traffic_counter_to[peer_name] = 0
            self.send_locks[peer_name] = threading.Lock()

        peers = [(peer_addr, self.other_addrs[peer_addr]) for peer_addr in self.other_addrs]
        parallel(connect_one, peers)
        while self.listening:
            time.sleep(0.1)
        return

    def send_to(self, name: str, data: bytes):
        self.send_locks[name].acquire()
        if name not in self.other_send_sockets:
            raise SocketException("Peer name %s dose not exist or not connected yet" % name)
        s = self.other_send_sockets[name]
        write_socket(s, data)
        self.traffic_counter_to[name] += len(data) + len_header
        self.send_locks[name].release()

    def recv_from(self, name):
        if name not in self.other_recv_sockets:
            raise SocketException("Peer name %s dose not exist or not connected yet" % name)
        s = self.other_recv_sockets[name]
        content = read_socket(s)
        self.traffic_counter_from[name] += len(content) + len_header
        return content

    def reset_counter(self):
        for k in self.traffic_counter_from:
            self.traffic_counter_from[k] = 0
        for k in self.traffic_counter_to:
            self.traffic_counter_to[k] = 0

    def terminate(self):
        self.socket.close()
        for peer_name in self.other_send_sockets:
            self.other_send_sockets[peer_name].close()
        for peer_name in self.other_recv_sockets:
            self.other_recv_sockets[peer_name].close()
