import pickle
import threading
from Amber.Core.Comm.Socket import SocketServer
import logging

logger = logging.getLogger("Peer")


class PeerException(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


class PackedMessage:
    def __init__(self, header: str, obj: object):
        self.header = header
        self.obj = obj

    def serialize(self):
        return pickle.dumps(self)


class Peer(SocketServer):
    def __init__(self, address: str, other_addrs: dict, timeout=120):
        super(Peer, self).__init__(address, other_addrs, timeout)
        self.prefetch_buffer = dict()
        self.recv_lock = dict()
        for other_name in other_addrs.values():
            self.prefetch_buffer[other_name] = dict()
            self.recv_lock[other_name] = threading.Lock()

    def send(self, peer_name: str, header: str, obj: object=None):
        self.send_to(peer_name, PackedMessage(header, obj).serialize())

    def recv(self, peer_name: str, header: str):
        self.recv_lock[peer_name].acquire()
        if header in self.prefetch_buffer.get(peer_name):
            packed_message = self.prefetch_buffer[peer_name].pop(header)
        else:
            packed_message = pickle.loads(self.recv_from(peer_name))
            if not isinstance(packed_message, PackedMessage):
                raise PeerException("Message corrupted or wrong message")

            while packed_message.header != header:
                if packed_message.header in self.prefetch_buffer[peer_name]:
                    raise PeerException(f"Already have message {header} in the buffer, cannot buffer >1 messages")
                self.prefetch_buffer[peer_name][packed_message.header] = packed_message
                packed_message = pickle.loads(self.recv_from(peer_name))
                if not isinstance(packed_message, PackedMessage):
                    raise PeerException("Message corrupted or wrong message")

        logger.debug("Received %s message from %s" % (header, peer_name))
        self.recv_lock[peer_name].release()
        return packed_message.obj

