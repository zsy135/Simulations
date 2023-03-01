import abc
import os


class Package:

    def __init__(self):
        self._format = None
        self._fresh = False
        self._data = None

    def pack(self):
        if self._fresh and self._data:
            return self._data
        else:
            self._fresh = True
            return self._pack()

    @abc.abstractmethod
    def _pack(self):
        pass

    @abc.abstractmethod
    def _unpack(self, data):
        pass

    def unpack(self, data):
        self._fresh = True
        return self._unpack(data)

    def print(self):
        print("-" * 100)
        self._print()
        print("-" * 100)

    @abc.abstractmethod
    def _print(self):
        pass

    @property
    def format(self):
        return self._format

    def save(self, path):
        with open(path, "wb") as f:
            data = self.pack()
            f.write(data)

    def load(self, path):
        if not os.path.exists(path):
            raise Exception("file path is not valid!!!")
        with open(path, "rb") as f:
            data = f.read()
            self.unpack(data)
