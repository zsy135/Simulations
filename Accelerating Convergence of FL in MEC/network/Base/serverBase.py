import abc
import socket
import threading
import queue
import time


class Server:

    def __init__(self, address_, port_):
        self._state = 0  # 0 : not active ； 1 ： active
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        if address_:
            self._host_address = address_
        else:
            self._host_address = "0.0.0.0"
        if port_ < 0:
            raise Exception("port is not valid!!!!")
        self._port = port_
        self._running = threading.Event()
        self._running.set()
        self._requests = queue.Queue()
        self._no_request_condition = threading.Condition()
        self._worker = None

    @property
    def state(self):
        return self._state

    @property
    def socket(self):
        return self._socket

    @property
    def host_address(self):
        return self._host_address

    def state2str(self):
        if self._state == 0:
            return "server is not active"
        elif self._state == 1:
            return "severing"
        else:
            return "无意义!!!"

    def reset(self):
        self._state = 0  # 0 : not active ； 1 ： active
        self.shutdown()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        self._running.set()
        self._requests.empty()
        self._reset()

    @abc.abstractmethod
    def _reset(self):
        pass

    @abc.abstractmethod
    def _before_active(self):
        pass

    def run(self):
        self._before_active()
        self._active()

    def _active(self):
        self._socket.bind((self._host_address, self._port))
        self._socket.listen()
        self._worker = threading.Thread(target=self._process_request)
        self._worker.start()
        self._server()

    def _server(self):
        while self._running.is_set():
            try:
                request = self._socket.accept()
                self._add_request(request)
            except OSError:
                pass

    def _add_request(self, r):
        self._requests.put(r)
        with self._no_request_condition:
            self._no_request_condition.notifyAll()

    def _process_request(self):
        while self._running.is_set():
            with self._no_request_condition:
                while self._requests.qsize() == 0 and self._running.is_set():
                    self._no_request_condition.wait()
            r = self._requests.get()
            self.handle_request(r)

    @abc.abstractmethod
    def handle_request(self, request):
        pass

    def shutdown(self):
        print("server stop running -----------------------------")
        self._running.clear()
        self._no_request_condition.notifyAll()
        try:
            self._socket.close()
        except OSError:
            pass
        self._shutdown()

    @abc.abstractmethod
    def _shutdown(self):
        pass


# test

if __name__ == "__main__":
    class MyServer(Server):

        def _before_active(self):
            pass

        def __init__(self, address_, port_):
            super(MyServer, self).__init__(address_, port_)

        def _shutdown(self):
            pass

        def _reset(self):
            pass

        def handle_request(self, request):
            s, address_ = request
            print(address_)
            s.shutdown()


    address = ("127.0.0.1", 10001)

    # client
    def fun():
        time.sleep(5)  # 等待服务器运行
        for i in range(100):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
            try:
                s.connect(address)
            except ConnectionError:
                time.sleep(0.5)  # 等待0.5秒后再尝试
            finally:
                s.close()


    t = threading.Thread(target=fun)
    t.start()

    # server
    server = MyServer(address[0], address[1])
    server.run()
