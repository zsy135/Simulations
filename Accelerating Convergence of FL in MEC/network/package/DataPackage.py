import math
import socket

from network.Base.packageBase import Package
import struct


class DataPack(Package):

    def __init__(self, bytes_):
        super(DataPack, self).__init__()
        self._bytes = bytes(bytes_)
        self._len = len(bytes_)

    _buffer_size = 256  # 接受数据的缓存大小
    _format = '<I'

    @classmethod
    def receive_data(cls, socket_: socket.socket):
        size_ = struct.calcsize(DataPack._format)
        len_ = socket_.recv(size_)
        len_ = struct.unpack(DataPack._format, len_)[0]
        times = math.ceil(len_ / DataPack._buffer_size)
        data_ = bytes()
        while True:
            data_ = data_ + socket_.recv(DataPack._buffer_size)
            if len(data_) == len_:
                break
        return data_

    @property
    def len(self):
        return self._len

    @len.setter
    def len(self, v):
        self._fresh = False
        self._len = v

    def _pack(self):
        len_b = struct.pack(DataPack._format, self._len)
        self._data = len_b + self._bytes
        return self._data

    def _unpack(self, data):
        self._len = struct.unpack_from(DataPack._format, data[0:4])
        self._data = bytes(data)
        self._bytes = data[4:]
        return self._len, self._bytes

    def __repr__(self):
        code_f = "len: {}\n".format(self._len)
        width = 16
        data_f = ""
        for i in range(self._len):
            data_f = data_f + hex(self._bytes[i]) + "  "
            if (i + 1) % width == 0:
                data_f = data_f + "\n"
        data_f = "data: \n" + data_f
        return code_f + data_f

    def _print(self):
        print(repr(self))
