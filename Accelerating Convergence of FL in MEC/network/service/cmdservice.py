import abc
import queue
import socket
import threading
import time

from network.package.CommandPackage import CmdPack
from utils.net import get_socket
from network.package.DataPackage import DataPack


class CmdService:
    class SendReq:
        def __init__(self, data_):
            self._finished = False
            self._data_ = data_

        @property
        def finished(self):
            return self._finished

        @finished.setter
        def finished(self, v):
            self._finished = v

        @property
        def data(self):
            return self._data_

    def __init__(self, socket_):
        self._socket: socket.socket = socket_
        self._remote_address = self._socket.getpeername()
        self._cmd_handle_table_mutex = threading.Lock()
        self._cmd_handle_table = dict()
        self._executing_cmd_event = threading.Event()  # 是否开启运行命令的功能
        self._sending_cmd_event = threading.Event()  # 是否开启发送命令的功能
        self._receive_cmd_event = threading.Event()  # 是否开启发送命令的功能
        self.receive_buffer_size = 1024
        self._send_blocking_cn = threading.Condition()
        self._un_send_data_queue = queue.Queue()
        self._send_worker = None
        self._send_worker_waiter = threading.Condition()
        self._init()

    def close(self):
        self.stop_send_cmd()
        self.stop_execute_cmd()
        self._socket.close()

    @property
    def socket(self):
        return self._socket

    @property
    def remote_address(self):
        return self._remote_address

    def _init(self):
        self._receive_cmd_event.set()
        self._executing_cmd_event.set()
        self._sending_cmd_event.set()
        self._set_default_handle()
        self._un_send_data_queue.empty()
        self._send_worker = threading.Thread(target=self._sender_target)
        self._send_worker.start()

    def _set_default_handle(self):
        handle = DefaultCmdHandle()
        for cmd_code_, _ in CmdPack.StringOfCmdTable.items():
            self.register_cmd_execute(cmd_code_, handle)

    def _send_data(self, s_req: SendReq):
        data_ = s_req.data
        try:
            self._socket.send(data_)
        except Exception as e:
            raise e
        else:
            self._send_finish(s_req)

    def _send_finish(self, s_req: SendReq):
        s_req.finished = True
        with self._send_blocking_cn:
            self._send_blocking_cn.notifyAll()

    def _sender_target(self):
        while self._sending_cmd_event.is_set():
            with self._send_worker_waiter:
                while self._un_send_data_queue.qsize() == 0 and self._sending_cmd_event.is_set():
                    self._send_worker_waiter.wait()
            try:
                req_ = self._un_send_data_queue.get(block=False)
            except queue.Empty:
                pass
            else:
                self._send_data(req_)

    def _send_cmd_blocking(self, s_req: SendReq):
        self._submit_send_req(s_req)
        with self._send_blocking_cn:
            while not s_req.finished:
                self._send_blocking_cn.wait()

    def _send_cmd_not_blocking(self, s_req: SendReq):
        return self._submit_send_req(s_req)

    def _submit_send_req(self, r):
        self._un_send_data_queue.put(r)
        with self._send_worker_waiter:
            self._send_worker_waiter.notifyAll()

    def send_cmd(self, cmd_code, arg, blocking=True):
        data_ = CmdPack(cmd_code, arg).pack()
        s_req = CmdService.SendReq(data_)
        if blocking:
            self._send_cmd_blocking(s_req)
        else:
            self._send_cmd_not_blocking(s_req)

    def execute_cmd(self, cmd_, *args):
        if self._executing_cmd_event.is_set():
            try:
                r = self._dispatch(cmd_, *args)
            except Exception as e:
                self._executing_cmd_event.clear()
                raise e
            else:
                return r

    def _dispatch(self, cmd_, *args):
        handle: CmdHandle = self.get_cmd_handle(cmd_.cmd_code)
        if callable(handle):
            return handle(self, cmd_, *args)
        else:
            raise Exception("Can`t get a handle !!!!")

    def stop_execute_cmd(self):
        self._executing_cmd_event.clear()

    def stop_send_cmd(self):
        self._sending_cmd_event.clear()
        with self._send_worker_waiter:
            self._send_worker_waiter.notifyAll()

    def receive_cmd(self) -> CmdPack:  # 该函数目前只能一个一个命令的接受，同时接受多个命令将产生难以预料的错误
        if self._receive_cmd_event.is_set():
            _error = False
            try:
                cmd_ = CmdPack.receive_cmd(self._socket)
            except ConnectionError:
                _error = True
            else:
                return cmd_
            if _error:
                self.close()
                exit(-1)

    def get_cmd_handle(self, cmd_code):
        self._cmd_handle_table_mutex.acquire()
        try:
            handler = self._cmd_handle_table[cmd_code]
        except KeyError:
            handler = None
        finally:
            self._cmd_handle_table_mutex.release()
        return handler

    def register_cmd_execute(self, cmd_code, handler, blocking=True):
        if blocking:
            self._cmd_handle_table_mutex.acquire()
            try:
                self._cmd_handle_table[cmd_code] = handler

            except KeyError:
                pass
            finally:
                self._cmd_handle_table_mutex.release()
            return True
        else:
            gotit = self._cmd_handle_table_mutex.acquire(blocking=False)
            if gotit:
                try:
                    self._cmd_handle_table[cmd_code] = handler

                except KeyError:
                    pass
                finally:
                    self._cmd_handle_table_mutex.release()
            return gotit


class CmdHandle:
    @abc.abstractmethod
    def before_handle(self, service: CmdService, *args):
        pass

    @abc.abstractmethod
    def handler(self, service: CmdService, *args):
        pass

    @abc.abstractmethod
    def after_handle(self, service: CmdService, r, *args):
        pass

    @classmethod
    def receive_data(cls, remote_address):
        s_ = get_socket()
        CmdHandle.try_connect(s_, remote_address)
        data_ = DataPack.receive_data(s_)
        s_.close()
        return data_

    @classmethod
    def receive_data_with_no_close_socket(cls, remote_address):
        s_ = get_socket()
        CmdHandle.try_connect(s_, remote_address)
        data_ = DataPack.receive_data(s_)
        return data_, s_

    @classmethod
    def send_data(cls, remote_address, data_):
        s_ = get_socket()
        try:
            CmdHandle.try_connect(s_, remote_address)
        except ConnectionError:
            s_.close()
        else:
            data_ = DataPack(data_)
            len_ = s_.send(data_.pack())
            s_.close()
            return len_

    @classmethod
    def try_connect(cls, s_: socket.socket, address_):
        times = 10  # 尝试建立连接10次
        for _ in range(times):
            try:
                s_.connect(address_)
            except ConnectionError:
                time.sleep(1.0)
            else:
                return s_
        raise ConnectionError("Data link connect fail !!")

    def __call__(self, service: CmdService, *args):
        self.before_handle(service, *args)
        try:
            r = self.handler(service, *args)
        except Exception as e:
            print("handle cmd ( ", args, "):  An ERROR Occur!!!!")
            raise e
        else:
            r = self.after_handle(service, r, *args)
            return r


class DefaultCmdHandle(CmdHandle):

    def before_handle(self, service: CmdService, *args):
        print("start deal the cmd.....")

    def handler(self, service: CmdService, *args):
        print("The Cmd: ")
        cmd_: CmdPack = args[0]
        cmd_.print()
        return None

    def after_handle(self, service: CmdService, r, *args):
        print("finish handling the cmd!!")
        return r
