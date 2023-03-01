import socket
import threading

from network.Base.serverBase import Server
from network.server.dt import DT
from utils.net import receive_data
from network.package.RequestPackage import RequestPack


class DtServer(Server):

    def __init__(self, address_, port_):
        super(DtServer, self).__init__(address_, port_)
        self._dts = dict()
        self._nextid = 1
        self._dt_access_mutex = threading.RLock()
        self._dt_generate_mutex = threading.RLock()
        self._dt_generate_one_condition = threading.Condition()  # 当申请dt时，如果此时没有空闲dt,则进程睡眠在测条件上，当有新的dt产生时，唤醒所有睡眠的线程
        self._allocated_dts = dict()
        self._unallocated_dts = dict()
        self._dt_numbers = 0

    @property
    def dts_dict(self):
        return self._dts

    def get_dt_nums(self):
        return self._dt_numbers

    def get_free_dt_nums(self):
        return len(self._unallocated_dts)

    def get_used_dt_nums(self):
        return len(self._allocated_dts)

    def delete_dt(self, id_):
        self._dt_access_mutex.acquire()
        try:
            self._unallocated_dts.pop(id_)
        except KeyError:
            pass
        try:
            self._allocated_dts.pop(id_)
        except KeyError:
            pass
        try:
            dt_: DT = self._dts.pop(id_)
        except KeyError:
            return True
        else:
            self._dt_numbers -= 1
            dt_.shutdown()
        finally:
            self._dt_access_mutex.release()
        return True

    def allocate_n_dt(self, n, blocking=True):
        dts = []
        for _ in range(n):
            dt_ = self.allocate_one_dt(blocking)
            if dt_:
                dts.append(dt_)
            else:
                break
        return dts

    def allocate_one_dt(self, blocking=True):
        if blocking:
            if self.get_free_dt_nums() <= 0:
                with self._dt_generate_one_condition:
                    self._dt_generate_one_condition.wait()
        self._dt_access_mutex.acquire()
        id_, dt_ = None, None
        try:
            for key_, v_ in self._unallocated_dts.items():
                id_, dt_ = key_, v_
                break
        except KeyError:
            pass
        else:
            self._unallocated_dts.pop(id_)
            self._allocated_dts[id_] = dt_
        finally:
            self._dt_access_mutex.release()
        return dt_

    def allocate_dt_by_id(self, id_):
        self._dt_access_mutex.acquire()
        try:
            dt_ = self._unallocated_dts[id_]
        except KeyError:
            dt_ = None
        else:
            self._unallocated_dts.pop(id_)
            self._allocated_dts[id_] = dt_
        finally:
            self._dt_access_mutex.release()
        return dt_

    def _generate_dt(self, client: socket.socket) -> DT:
        self._dt_generate_mutex.acquire()
        id_ = self._nextid
        try:
            dt_ = DT(client, self, id_)
        except Exception:
            raise Exception("Generate dt fail !!")
        else:
            self._nextid += 1
            self._dt_numbers += 1
            return dt_
        finally:
            self._dt_generate_mutex.release()

    def _reset(self):
        self.shutdown()
        self._dts.clear()
        self._allocated_dts.clear()
        self._unallocated_dts.clear()
        self._dt_numbers = 0
        self._nextid = 0

    def _before_active(self):
        print("Server starting ......")

    def handle_request(self, request):
        client: socket.socket = request[0]
        data = receive_data(client)
        try:
            r = self._dispatch_request(client, data)
        except Exception as e:
            raise e
        else:
            return r

    def _dispatch_request(self, client, data):
        requestPack = RequestPack()
        requestPack.unpack(data)
        if requestPack.code == 1:
            return self._process_request_1(client, requestPack)
        elif requestPack.code == 2:
            return self._process_request_2(client, requestPack)
        elif requestPack.code == 5:
            return self._process_request_5(client)
        else:
            raise Exception("Can not deal unknown request !!")

    def _process_request_1(self, client: socket.socket, req: RequestPack):
        id_ = req.id
        dt = None
        if id_ == 0:
            dt = self._generate_dt(client)
            self._add_dt(dt)
            ret_pack = RequestPack(3, dt.id)
            client.send(ret_pack.pack())
            self._after_process_request_1(dt)
        else:
            pass
        return dt

    def _after_process_request_1(self, dt: DT):
        with self._dt_generate_one_condition:
            self._dt_generate_one_condition.notifyAll()
        print("** Generate A DT:")
        print("        Id: ", dt.id)
        print("        Address: ", dt.socket.getsockname())
        print("        Remote Address: ", dt.remote_address)

    def _add_dt(self, dt_: DT):
        self._dt_access_mutex.acquire()
        self._dts[dt_.id] = dt_
        self._unallocated_dts[dt_.id] = dt_
        self._dt_access_mutex.release()

    def _process_request_2(self, client: socket.socket, req: RequestPack):
        id_ = req.id
        if id_ != 0:
            r = self.delete_dt(id_)
            client.close()
            return r
        client.close()
        return False

    def _process_request_5(self, client: socket.socket):
        ret_pack = RequestPack(6, self._dt_numbers)
        client.send(ret_pack.pack())
        client.close()

    def _shutdown(self):
        self._dt_access_mutex.acquire()
        for id_, dt_ in self._dts.items():
            dt__: DT = dt_
            dt__.shutdown()
        self._dt_access_mutex.release()
