import socket

from network.Base.packageBase import Package
import struct


class CmdPack(Package):
    """
        添加新的命令时，在写该命令的参数打包函数时，其命名方式为： _pack_cmd_cmdvalue(),例如对于命令3，函数可命名为
             _pack_cmd_3()；
        同理，对于解包参数的函数命名，_pack_cmd_cmdvalue，对于命令3，有：_parse_cmd_3()；
        同时，需要在cmd2str_table和cmd2str_table两个表中添加对应项。

    """
    Config_Format = "json"

    ArgOffside = 3
    StringOfCmdTable = {
        0: "stop", 1: "start testing", 2: "start training", 4: "send dataset", 5: "update MPL weights",
        6: "get MPL weights", 7: "upload MPL weights", 8: "upload MPL weights & suspend", 9: "continue",
        10: "config", 11: "list config", 66: "broadcasting state data", 67: "send broadcasting state data",
        68: "request actions", 69: "return for request actions",
        125: "keep alive", 126: "exception", 127: "return"
    }
    CmdOfStringTable = {'stop': 0, 'start testing': 1, 'start training': 2, 'send dataset': 4, 'update MPL weights': 5,
                        'get MPL weights': 6, 'upload MPL weights': 7, 'upload MPL weights & suspend': 8, 'continue': 9,
                        'config': 10, 'list config': 11, 'broadcasting state data': 66,
                        'send broadcasting state data': 67, "request actions": 68, "return for request actions": 69,
                        'keep alive': 125, 'exception': 126, 'return': 127
                        }
    CmdArgFormatTable = {
        0: "", 1: "", 2: "", 4: "<H", 5: "<H",
        6: "<H", 7: "<H", 8: "<H", 9: "",
        10: "<H6B", 11: "<H6B", 66: "<H", 67: "<HI",
        68: "<H", 69: "<HI",
        125: "<I", 126: "<H", 127: None
    }
    CmdFormat = "<B"
    LenFormat = "<H"
    CmdSize = struct.calcsize(CmdFormat)
    LenSize = struct.calcsize(LenFormat)

    _cmd_code_l = 0  # 命令码的最小值
    _cmd_code_h = 127  # 命令码的最大值

    def __init__(self, cmd_=0, arg_=None):
        super(CmdPack, self).__init__()
        self._cmd = cmd_
        self._len = CmdPack.get_arg_len(cmd_)
        if self._len is None:
            self._len = len(arg_)
        self._arg = arg_

    @classmethod
    def get_arg_len(cls, c_):  # 获得某个命令的参数长度，对于没定义的命令，返回None
        _f = CmdPack.CmdArgFormatTable[c_]
        if _f is None:
            return None
        return struct.calcsize(_f)

    @classmethod
    def receive_cmd(self, s_: socket.socket):
        data_code = s_.recv(CmdPack.CmdSize)
        data_len = s_.recv(CmdPack.LenSize)
        len_ = struct.unpack(CmdPack.LenFormat, data_len)[0]
        data_arg = s_.recv(len_)
        data_ = data_code + data_len + data_arg
        cmd_ = CmdPack()
        cmd_.unpack(data_)
        return cmd_

    @property
    def cmd_code(self):
        return self._cmd

    @cmd_code.setter
    def cmd_code(self, v):
        self._fresh = False
        self._cmd = v

    @property
    def len(self):
        return self._len

    @len.setter
    def len(self, v):
        self._fresh = False
        self._len = v

    @property
    def arg(self):
        return self._arg

    @arg.setter
    def arg(self, v):
        self._fresh = False
        self._arg = v

    @classmethod
    def check(cls, code_):
        if cls._cmd_code_h >= code_ >= cls._cmd_code_l:
            return True
        else:
            return False

    def is_start_train(self):
        if self._cmd == 2:
            return True
        return False

    def is_update_weights(self):
        if self._cmd == 5:
            return True
        return False

    def is_upload_weights(self):
        if self._cmd == 7:
            return True
        return False

    def is_broadcasting_state_data(self):
        if self._cmd == 66:
            return True
        return False

    def is_send_broadcasting_state_data(self):
        if self._cmd == 67:
            return True
        return False

    def is_request_action(self):
        if self._cmd == 68:
            return True
        return False

    def is_return_for_request_actions(self):
        if self._cmd == 69:
            return True
        return False

    def _pack(self):
        cmd_b = struct.pack(CmdPack.CmdFormat, self._cmd)
        len_b = struct.pack(CmdPack.LenFormat, self._len)
        self._data = cmd_b + len_b + self._pack_arg()
        return self._data

    def _unpack(self, data):
        s_ = struct.calcsize(CmdPack.CmdFormat)
        self._cmd = struct.unpack_from(CmdPack.CmdFormat, data[0:s_], 0)[0]
        self._len = CmdPack.get_arg_len(self._cmd)
        arg_ = data[CmdPack.ArgOffside:]
        self._arg = self.parse_arg(self._cmd, arg_)
        self._data = bytes(data)
        return self._cmd, self._len, self._arg

    @classmethod
    def parse_arg(cls, cmd_, arg_):
        handle_name = "_parse_cmd_" + str(cmd_)
        try:
            handle = getattr(cls, handle_name)
        except AttributeError as e:
            print(e)
            raise e
        else:
            return handle(cmd_, arg_)

    @classmethod
    def _parse_cmd_no_arg(cls):
        return None

    @classmethod
    def _parse_cmd_0(cls, cmd_, arg_):
        return CmdPack._parse_cmd_no_arg()

    @classmethod
    def _parse_cmd_1(cls, cmd_, arg_):
        return CmdPack._parse_cmd_no_arg()

    @classmethod
    def _parse_cmd_2(cls, cmd_, arg_):
        return CmdPack._parse_cmd_no_arg()

    @classmethod
    def _parse_cmd_short(cls, cmd_, arg_):
        return struct.unpack_from("<H", arg_[0:2])[0]

    @classmethod
    def _parse_cmd_un_int(cls, cmd_, arg_):
        return struct.unpack_from("<I", arg_[0:4])[0]

    @classmethod
    def _parse_cmd_4(cls, cmd_, arg_):
        return CmdPack._parse_cmd_short(cmd_, arg_)

    @classmethod
    def _parse_cmd_5(cls, cmd_, arg_):
        return CmdPack._parse_cmd_short(cmd_, arg_)

    @classmethod
    def _parse_cmd_6(cls, cmd_, arg_):
        return CmdPack._parse_cmd_short(cmd_, arg_)

    @classmethod
    def _parse_cmd_7(cls, cmd_, arg_):
        return CmdPack._parse_cmd_short(cmd_, arg_)

    @classmethod
    def _parse_cmd_8(cls, cmd_, arg_):
        return CmdPack._parse_cmd_short(cmd_, arg_)

    @classmethod
    def _parse_cmd_9(cls, cmd_, arg_):
        return CmdPack._parse_cmd_no_arg()

    @classmethod
    def _parse_cmd_10(cls, cmd_, arg_):
        port_b = struct.unpack("<H", arg_[0:2])[0]
        format_: str = arg_[2:].decode()
        return port_b, format_

    @classmethod
    def _parse_cmd_11(cls, cmd_, arg_):
        port_b = struct.unpack("<H", arg_[0:2])[0]
        format_: str = arg_[2:8].decode()
        return port_b, format_

    @classmethod
    def _parse_cmd_66(cls, cmd_, arg_):
        return CmdPack._parse_cmd_short(cmd_, arg_)

    @classmethod
    def _parse_cmd_67(cls, cmd_, arg_):
        port_ = struct.unpack("<H", arg_[0:2])[0]
        id_ = struct.unpack("<I", arg_[2:])[0]
        return port_, id_

    @classmethod
    def _parse_cmd_68(cls, cmd_, arg_):
        return CmdPack._parse_cmd_short(cmd_, arg_)

    @classmethod
    def _parse_cmd_69(cls, cmd_, arg_):
        port_ = struct.unpack("<H", arg_[0:2])[0]
        id_ = struct.unpack("<I", arg_[2:])[0]
        return port_, id_

    @classmethod
    def _parse_cmd_125(cls, cmd_, arg_):
        return CmdPack._parse_cmd_short(cmd_, arg_)

    @classmethod
    def _parse_cmd_126(cls, cmd_, arg_):
        return CmdPack._parse_cmd_short(cmd_, arg_)

    @classmethod
    def _parse_cmd_127(cls, cmd_, arg_):
        return arg_

    def _pack_arg(self):
        handle_name = "_pack_cmd_" + str(self._cmd)
        try:
            handle = getattr(self, handle_name)
        except AttributeError as e:
            print(e)
            raise e
        else:
            return handle()

    def _pack_cmd_no_arg(self):
        if self._len == 0:
            return bytes()
        else:
            return bytes()

    def _pack_cmd_0(self):
        return self._pack_cmd_no_arg()

    def _pack_cmd_1(self):
        return self._pack_cmd_no_arg()

    def _pack_cmd_2(self):
        return self._pack_cmd_no_arg()

    def _pack_cmd_short(self):
        return struct.pack("<H", self._arg)

    def _pack_cmd_un_int(self):
        return struct.pack("<I", self._arg)

    def _pack_cmd_4(self):
        return self._pack_cmd_short()

    def _pack_cmd_5(self):
        return self._pack_cmd_short()

    def _pack_cmd_6(self):
        return self._pack_cmd_short()

    def _pack_cmd_7(self):
        return self._pack_cmd_short()

    def _pack_cmd_8(self):
        return self._pack_cmd_short()

    def _pack_cmd_9(self):
        return self._pack_cmd_no_arg()

    def _pack_cmd_10(self):
        _p, _f = self._arg
        port_b = struct.pack("<H", _p)
        format_: str = _f
        format_b = format_.encode()
        return port_b + format_b

    def _pack_cmd_11(self):
        _p, _f = self._arg
        port_b = struct.pack("<H", _p)
        format_: str = _f
        format_b = format_.encode()
        return port_b + format_b

    def _pack_cmd_66(self):
        return self._pack_cmd_short()

    def _pack_cmd_67(self):
        port_, id_ = self._arg
        port_b = struct.pack("<H", port_)
        id_b = struct.pack("<I", id_)
        return port_b + id_b

    def _pack_cmd_68(self):
        return self._pack_cmd_short()

    def _pack_cmd_69(self):
        port_, id_ = self._arg
        port_b = struct.pack("<H", port_)
        id_b = struct.pack("<I", id_)
        return port_b + id_b

    def _pack_cmd_125(self):
        return self._pack_cmd_un_int()

    def _pack_cmd_126(self):
        return self._pack_cmd_short()

    def _pack_cmd_127(self):
        return self._arg

    @classmethod
    def cmd2str(cls, cmd):
        try:
            t = CmdPack.StringOfCmdTable[cmd]
        except KeyError as e:
            return "无意义"
        else:
            return t

    @classmethod
    def str2cmd(cls, s):
        try:
            t = CmdPack.StringOfCmdTable[s]
        except KeyError as e:
            return "无意义"
        else:
            return t

    def __repr__(self):
        cmd_f = "code: {:<20d}     ({})\n".format(self._cmd, self.cmd2str(self._cmd))
        len_f = "len: {}\n".format(self._len)
        arg_f = "arg: {}\n".format(self._arg)
        return cmd_f + len_f + arg_f

    def _print(self):
        print(repr(self))
