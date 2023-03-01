from network.Base.packageBase import Package
import struct


class RequestPack(Package):

    def __init__(self, code=0, id_=0):
        super(RequestPack, self).__init__()
        self._code = code
        self._id = id_
        self._padding = 0
        self._format = '<BHB'
        self._fresh = False
        self._data = None

    def is_register_req(self):
        if self.code == 1:
            return True

    def is_unregister_req(self):
        if self.code == 2:
            return True

    def is_register_successful(self):
        if self._code == 3:
            return True

    def is_register_unsuccessful(self):
        if self._code == 4:
            return True

    def is_request_dt_number(self):
        if self._code == 5:
            return True

    def is_return_of_request_dt_number(self):
        if self._code == 6:
            return True

    @property
    def code(self):
        return self._code

    @code.setter
    def code(self, v):
        self._fresh = False
        self._code = v

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, v):
        self._fresh = False
        self._id = v

    def _pack(self):
        self._data = struct.pack(self._format, self._code, self._id, self._padding)
        return self._data

    def _unpack(self, data_):
        code_, id_, _ = struct.unpack(self._format, data_)
        self._code = code_
        self._id = id_
        self._data = bytes(data_)
        return code_, id_

    @classmethod
    def code2str(cls, code):
        if code == 0:
            return "格式错误"
        elif code == 1:
            return "注册请求"
        elif code == 2:
            return "注销请求"
        elif code == 3:
            return "注册成功"
        elif code == 4:
            return "注册失败"
        elif code == 5:
            return "请求数目"
        elif code == 6:
            return "回复：请求数目"
        else:
            return "无意义"

    def __repr__(self):
        code_f = "code: {}     ({:^20s})\n".format(self._code, self.code2str(self._code))
        id_f = "id: {}\n".format(self._id)
        return code_f + id_f

    def _print(self):
        print(repr(self))


if __name__ == "__main__":
    req_ = RequestPack(1, 2)
    data = req_.pack()
    un_req = RequestPack()
    un_req.unpack(data)
