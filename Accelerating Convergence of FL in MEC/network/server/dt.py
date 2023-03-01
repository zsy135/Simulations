import pickle
import queue
import random
import socket
import struct

import utils.model
from network.Base.dtBase import DTBase

from network.service.cmdservice import CmdService
from network.service.cmdservice import CmdHandle
from network.package.CommandPackage import CmdPack
from utils.net import get_socket
from network.package.DataPackage import DataPack
from utils.model import parse_config
from utils.model import pack_config


class Cmd7Handle(CmdHandle):

    def before_handle(self, service: CmdService, *args):
        cmd_: CmdPack = args[0]
        # print("Start dealing Cmd from " + str(cmd_.cmd_code) + ".....")
        # cmd_.print()

    def handler(self, service: CmdService, *args):
        s_: socket.socket = get_socket()
        if not s_:
            return False, None
        cmd_: CmdPack = args[0]
        p_ = cmd_.arg
        address_ = (service.remote_address[0], p_)
        s_ = Cmd7Handle.try_connect(s_, address_)
        data = DataPack.receive_data(s_)
        s_.close()
        return True, data

    def after_handle(self, service: CmdService, r, *args):  # 目前该函数功能比较简单，后面可以考虑增加命令返回确认功能
        dealt_, d_ = r
        cmd_: CmdPack = args[0]
        if dealt_:
            # 后面需要对收到的数据进行处理后再返回
            # print("Cmd: ", CmdPack.StringOfCmdTable[cmd_.cmd_code], "is dealt successfully!")
            return d_
        else:
            # print("Cmd: ", CmdPack.StringOfCmdTable[cmd_.cmd_code], "is not dealt successfully!")
            return None


class Cmd8Handle(Cmd7Handle):

    def after_handle(self, service: CmdService, r, *args):
        d_ = super().after_handle(service, r, *args)
        if d_:
            # return continue cmd
            service.send_cmd(CmdPack.CmdOfStringTable['continue'], None, blocking=True)
        else:
            service.send_cmd(CmdPack.CmdOfStringTable['exception'], 4, blocking=True)
        return d_


class ReceiveCmdHandle(CmdHandle):

    def before_handle(self, service: CmdService, *args):
        cmd_: CmdPack = args[0]
        # print("Start dealing Cmd from " + str(cmd_.cmd_code) + ".....")
        # print(repr(cmd_))

    def handler(self, service: CmdService, *args):
        cmd_: CmdPack = args[0]
        port_ = cmd_.arg
        ip_ = service.remote_address[0]
        try:
            data_ = self.receive_data((ip_, port_))
        except ConnectionError:
            return False
        else:
            return data_

    def after_handle(self, service: CmdService, r, *args):
        # print("Receive Data: ", len(r), "Bytes !!")
        cmd_: CmdPack = args[0]
        # print("Finish dealing Cmd " + str(cmd_.cmd_code) + ".....")
        return r


class Cmd127Handle(CmdHandle):

    def before_handle(self, service: CmdService, *args):
        pass

    def handler(self, service: CmdService, *args):
        cmd_: CmdPack = args[0]
        r = struct.unpack("<I", cmd_.arg[0:4])
        return r

    def after_handle(self, service: CmdService, r, *args):
        return r


class DT(DTBase, CmdService):
    def __init__(self, socket_, server_, id_):
        DTBase.__init__(self)
        CmdService.__init__(self, socket_)
        self._server = server_  # 辅助广播命令的执行
        self._set_cmd_handle()
        self._id = id_
        self._allocated = False
        self._cmd_un_deal_queue = queue.Queue()

    @property
    def allocated(self):
        return self._allocated

    @allocated.setter
    def allocated(self, v: bool):
        self._allocated = v

    @property
    def id(self):
        return self._id

    def _set_cmd_handle(self):
        handle_7 = Cmd7Handle()
        self.register_cmd_execute(CmdPack.CmdOfStringTable['upload MPL weights'], handle_7)

        handle_8 = Cmd8Handle()
        self.register_cmd_execute(CmdPack.CmdOfStringTable['upload MPL weights & suspend'], handle_8)

        handle_66 = ReceiveCmdHandle()
        self.register_cmd_execute(CmdPack.CmdOfStringTable['broadcasting state data'], handle_66)

        handle_68 = ReceiveCmdHandle()
        self.register_cmd_execute(CmdPack.CmdOfStringTable['request actions'], handle_68)
        return True

    def get_data_socket(self):
        s = get_socket()
        address_ip = self.socket.getsockname()[0]
        s.bind((address_ip, 0))
        address_ip, port = s.getsockname()
        return s, address_ip, port

    def _proxy_cmd_with_sending(self, s_, cmd_str, arg, data, blocking=True):
        self.send_cmd(CmdPack.CmdOfStringTable[cmd_str], arg, blocking)
        c, address_ = s_.accept()
        data = DataPack(data).pack()
        c.send(data)
        c.close()
        return True

    def _proxy_cmd_with_send_and_receive(self, s_, cmd_str, arg, data, blocking=True):
        self.send_cmd(CmdPack.CmdOfStringTable[cmd_str], arg, blocking)
        c, address_ = s_.accept()
        data = DataPack(data).pack()
        c.send(data)
        r_data = DataPack.receive_data(c)
        c.close()
        return r_data

    def _proxy_cmd_with_receive(self, s_, cmd_str, arg, blocking=True):
        self.send_cmd(CmdPack.CmdOfStringTable[cmd_str], arg, blocking)
        c, address_ = s_.accept()
        data = DataPack.receive_data(c)
        c.close()
        return data

    def get_weights(self):
        s_, address_ip, port_ = self.get_data_socket()
        s_.listen()
        r_ = self._proxy_cmd_with_receive(s_, "get MPL weights", port_)
        # print("Get Weights: ")
        # print(r_)
        s_.close()
        return r_

    def update_weights(self, array_):
        s_, address_ip, port_ = self.get_data_socket()
        s_.listen()
        data = pickle.dumps(array_)
        r_ = self._proxy_cmd_with_sending(s_, "update MPL weights", port_, data)
        s_.close()
        return r_

    def start_receive_cmd(self):
        while True:
            cmd_ = self.receive_cmd()
            if not self._try_deal_cmd(cmd_):
                self._cmd_un_deal_queue.put(cmd_)

    def _try_deal_cmd(self, cmd_: CmdPack):
        print("Cmd From ", self.id, "------------------------------------")
        if cmd_.is_broadcasting_state_data():
            return self._deal_experience_broadcasting(cmd_)
        elif cmd_.is_request_action():
            return self._deal_action_broadcasting(cmd_)
        else:
            return False

    def _deal_action_broadcasting(self, cmd_: CmdPack):
        data_ = self.execute_cmd(cmd_)
        for id_, dt_ in self._server.dts_dict.items():
            if id_ == self._id:
                continue
            d_: DT = dt_
            print("Send Req to DT: ", id_)
            r_ = d_.send_request_actions(data_)
            print("Send Return for Req !!")
            self.send_return_for_request_actions(r_)
            print("Finish Send Return for Req !!")
        return True

    def _deal_experience_broadcasting(self, cmd_: CmdPack):
        data_ = self.execute_cmd(cmd_)
        exp = utils.model.unpack_experience(data_)
        # print(exp)
        for id_, dt_ in self._server.dts_dict.items():
            if id_ == self._id:
                continue
            d_: DT = dt_
            d_.send_experience_broadcasting(data_)
        return True

    def send_experience_broadcasting(self, data_):
        s_, address_ip, port_ = self.get_data_socket()
        s_.listen()
        r_ = self._proxy_cmd_with_sending(s_, "send broadcasting state data", (port_, self.id), data_)
        s_.close()
        return r_

    def send_request_actions(self, data_):
        s_, address_ip, port_ = self.get_data_socket()
        s_.listen()
        r_ = self._proxy_cmd_with_send_and_receive(s_, "request actions", port_, data_)
        s_.close()
        return r_

    def send_return_for_request_actions(self, data_):
        s_, address_ip, port_ = self.get_data_socket()
        s_.listen()
        r_ = self._proxy_cmd_with_sending(s_, "return for request actions", (port_, self.id), data_)
        s_.close()
        return r_

    def get_cmd(self):
        return self._cmd_un_deal_queue.get()

    def send_dataset(self, data):
        pass

    def start_train(self):
        r = self.send_cmd(CmdPack.CmdOfStringTable['start training'], 0)
        return r

    def start_test(self):
        r = self.send_cmd(CmdPack.CmdOfStringTable['start testing'], 0)
        return r

    def predict(self):
        pass

    def list_configs(self):
        s_, address_ip, port_ = self.get_data_socket()
        s_.listen()
        r_ = self._proxy_cmd_with_receive(s_, "list config", port_)
        config_ = parse_config(r_, CmdPack.Config_Format)
        s_.close()
        return config_

    def update_configs(self, config):
        data = pack_config(config, CmdPack.Config_Format)
        s_, address_ip, port_ = self.get_data_socket()
        s_.listen()
        r_ = self._proxy_cmd_with_sending(s_, "config", port_, data)
        s_.close()
        return r_

    def check(self):
        arg_ = random.randint(10000, 9999999)
        try:
            self.send_cmd(CmdPack.CmdOfStringTable['keep alive'], arg_)
            cmd_ = self.receive_cmd()
        except TypeError:
            return False
        else:
            if cmd_.cmd_code == CmdPack.CmdOfStringTable["return"]:
                ret_ = self.execute_cmd(cmd_, )
                if ret_ == arg_:
                    return True
                else:
                    return False
            else:
                # 这里由于接受了其它的命令，导致命令传送中断，故抛出异常；以后通过升级彻底解决该问题
                raise Exception("not receive a return !!")

    def prepare_train(self):
        pass

    def shutdown(self):
        self.close()
