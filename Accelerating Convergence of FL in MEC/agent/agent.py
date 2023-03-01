import queue
import shutil
import socket
import threading
import time
import pickle

import numpy as np
import tensorflow as tf

import network.package.DataPackage
from network.service.cmdservice import CmdService
from network.service.cmdservice import CmdHandle
from network.package.CommandPackage import CmdPack
from utils.net import receive_data
from utils.net import get_socket
from agent.config import Config
import agent.config
from utils.model import format_write_dict
from network.package.RequestPackage import RequestPack
from agent.model.ddpg import DDPG
from agent.model.mlp import MLP
from agent.model.train import Train
from utils.model import pack_experience
from utils.model import unpack_experience
from utils.model import pack_state
from utils.model import unpack_states
from utils.model import pack_action
from utils.model import unpack_actions
from utils.net import get_data_socket
from agent.config import Config


class Agent:

    def __init__(self, _address="", _port=0):
        self._socket = get_socket()
        self._dt_id = -1
        if _address == "":
            self._address_ip = socket.gethostbyname(socket.gethostname())
        else:
            self._address_ip = _address
        self.session: tf.Session = tf.Session()
        self._dt_number = Config["number_of_agent"]
        self._port = _port
        self._server_address = (Config["Server_Ip"], Config["Server_Port"])
        self._cmd_service = None
        self._model_mlp_mutex = threading.RLock()
        self._model_ddpg_mutex = threading.RLock()
        self._train_epoch = 0
        self._wait_for_update_cmd = threading.Condition()
        self._update_weights = None
        self._cmd_un_deal_queue = queue.Queue()
        self._other_experience_queue = queue.Queue()
        self._other_actions_queue = queue.Queue()
        self._cmd_deal_worker = None
        self._is_training = False
        self.deal_cmd_in_wait_running = False
        self._trainer = None
        self.request_action_times_the_epoch_ = 1
        self.request_actions_finish_condition = threading.Condition()
        self.mlp = MLP()
        self.ddpg = DDPG()
        self._init()
        self.set_cmd_handle()

    @property
    def dt_number(self):
        return self._dt_number

    @property
    def weights_for_update(self):
        r_ = self._update_weights
        self._update_weights = None
        return r_

    @weights_for_update.setter
    def weights_for_update(self, v_):
        self._update_weights = v_

    @property
    def id(self):
        return self._dt_id

    @property
    def cmd_service(self) -> CmdService:
        return self._cmd_service

    def set_cmd_handle(self):
        handle5 = Cmd5Handle()
        self.cmd_service.register_cmd_execute(5, handle5)

        handle6 = Cmd6Handle()
        self.cmd_service.register_cmd_execute(6, handle6)

        handle67 = ReceiveCmdWithIdHandle()
        self.cmd_service.register_cmd_execute(67, handle67)

        handle68 = Cmd68Handle()
        self.cmd_service.register_cmd_execute(68, handle68)

        handle69 = ReceiveCmdWithIdHandle()
        self.cmd_service.register_cmd_execute(69, handle69)

    @property
    def server_address(self):
        return self._server_address

    @property
    def train_epoch(self):
        return self._train_epoch

    @property
    def port(self):
        return self._port

    @property
    def address_ip(self):
        return self._address_ip

    def _init(self):
        init_ = tf.global_variables_initializer()
        self.session.run(init_)
        self._try_connect_server()
        self._cmd_un_deal_queue.empty()

    def _try_connect_server(self):
        if not self._socket:
            raise Exception("socket is not valid !!")
        self._socket.bind((self._address_ip, self._port))
        self._port = self._socket.getsockname()[1]
        _connected = False
        for _ in range(10):
            try:
                self._socket.connect(self._server_address)
            except ConnectionRefusedError:
                time.sleep(0.5)
            else:
                if self._request_to_build_dt():
                    self._cmd_service = CmdService(self._socket)
                    _connected = True
                else:
                    raise Exception("Build dt fail !!")
                break
        if not _connected:
            raise ConnectionRefusedError()

    def upload_weights(self):
        weights = self.get_weights()
        if weights:
            self._upload_weights(weights)
        else:
            print("Don`t Get Weights!!!!")

    def get_weights(self):
        self._model_mlp_mutex.acquire()
        weights_ = self.mlp.get_weights(self.session)
        self._model_mlp_mutex.release()
        data_ = pickle.dumps(weights_)
        return data_

    def _upload_weights(self, weights):  # 实际做上传参数工作的函数
        s_, _, port_ = get_data_socket(self._address_ip)
        s_.listen()
        self._proxy_cmd_with_sending(s_, "upload MPL weights", port_, weights)
        s_.close()
        return weights

    @classmethod
    def update_config(cls, configs: dict):
        path = agent.config.__file__
        for k, v in configs:
            Config[k] = v
        path_bck = path + ".bck"
        shutil.copy(path, path_bck)
        try:
            format_write_dict(path, Config)
        except IOError:
            shutil.copy(path_bck, path)

    def _request_to_build_dt(self):
        req_ = RequestPack(1, 0)
        try:
            self._socket.send(req_.pack())
            data = receive_data(self._socket)
        except ConnectionError:
            return False
        else:
            ret_ = RequestPack()
            ret_.unpack(data)
            if ret_.is_register_successful():
                self._dt_id = ret_.id
                return True
            else:
                return False

    def update_weights(self, data_: bytes):
        weights_ = pickle.loads(data_)
        # print(weights_)
        self._model_mlp_mutex.acquire()
        self.mlp.set_weights(weights_, self.session)
        self._model_mlp_mutex.release()

    def _open_trainer(self):
        self._trainer = Train(self)

    def _deal_cmd(self):
        while True:
            if not self._is_training:
                cmd_: CmdPack = self._cmd_un_deal_queue.get()
                if cmd_.is_start_train():
                    self._open_trainer()
                    self._open_trainer()
                    self._is_training = True
                    self._train_step()
                else:
                    if not self._try_deal_cmd(cmd_):
                        self._execute_cmd(cmd_)
            else:
                try:
                    cmd_: CmdPack = self._cmd_un_deal_queue.get(block=False)
                except queue.Empty:
                    self._train_step()
                else:
                    if not cmd_.is_start_train():
                        if not self._try_deal_cmd(cmd_):
                            self._execute_cmd(cmd_)

    def _execute_cmd(self, cmd_):
        _cmd_service: CmdService = self._cmd_service
        r_ = _cmd_service.execute_cmd(cmd_, self)
        return r_

    def _train_step(self):
        self._train_epoch += 1
        if self._train_epoch == 200:
            self._trainer.finish_train()
            self._is_training = False
        if self._train_epoch % 5 == 0:
            self.save_model()
        trainer: Train = self._trainer
        trainer.step()

    def save_model(self):
        saver = tf.train.Saver()
        saver.save(self.session, "./Model_Save")

    def _start_deal_cmd_work(self):
        self._cmd_deal_worker = threading.Thread(target=self._receive_cmd_service)
        self._cmd_deal_worker.start()

    def _receive_cmd_service(self):
        service: CmdService = self._cmd_service
        while True:
            cmd_ = service.receive_cmd()
            self._cmd_un_deal_queue.put(cmd_)

    def run(self):
        self._start_deal_cmd_work()
        self._deal_cmd()

    def _try_deal_cmd(self, cmd_):
        if cmd_.is_update_weights():
            return self._trigger_update_weights_event(cmd_)
        elif cmd_.is_send_broadcasting_state_data():
            return self._deal_experience_broadcasting(cmd_)
        elif cmd_.is_return_for_request_actions():
            return self._deal_return_action_broadcasting(cmd_)
        return False

    def _deal_return_action_broadcasting(self, cmd_: CmdPack):
        # print ("Receive Return for Req!! ")
        action_ = self._execute_cmd(cmd_)
        self._other_actions_queue.put(action_)
        return True

    def _deal_experience_broadcasting(self, cmd_: CmdPack):
        exp_ = self._execute_cmd(cmd_)
        self._other_experience_queue.put(exp_)
        return True

    def deal_cmd_in_wait_exp_action(self):
        while self.deal_cmd_in_wait_running:
            self.do_deal_cmd_in_wait_exp_action()

    def do_deal_cmd_in_wait_exp_action(self):
        try:
            cmd_: CmdPack = self._cmd_un_deal_queue.get(block=False)
        except queue.Empty:
            pass
        else:
            if not cmd_.is_start_train():
                if not self._try_deal_cmd(cmd_):
                    self._execute_cmd(cmd_)

    def deal_action_request_cmd(self):
        try:
            cmd_: CmdPack = self._cmd_un_deal_queue.get(block=False)
        except queue.Empty:
            pass
        else:
            if cmd_.is_request_action():
                self._execute_cmd(cmd_)
            else:
                self._cmd_un_deal_queue.put(cmd_)

    def _trigger_update_weights_event(self, cmd_):
        self.weights_for_update = self._execute_cmd(cmd_)
        with self._wait_for_update_cmd:
            self._wait_for_update_cmd.notifyAll()
        return True

    def get_dt_number(self):
        s_ = get_socket()
        try:
            s_.connect(self._server_address)
        except ConnectionError as e:
            raise e
        else:
            req_ = RequestPack(5, 0)
            s_.send(req_.pack())
            data_ = s_.recv(1024)
            req_.unpack(data_)
            self._dt_number = req_.id
        finally:
            s_.close()
        return self._dt_number

    def fed_learn_once(self):
        self.upload_weights()
        self.deal_cmd_in_wait_running = True
        threading.Thread(target=self.fed_learn_once_target).start()
        self.deal_cmd_in_wait_exp_action()
        w_ = self.weights_for_update
        self.update_weights(w_)

    def fed_learn_once_target(self):
        while self._update_weights is None:
            with self._wait_for_update_cmd:
                self._wait_for_update_cmd.wait()
        self.deal_cmd_in_wait_running = False

    def get_other_agent_experience(self):
        threading.Thread(target=self.get_other_agent_experience_target).start()
        self.deal_cmd_in_wait_running = True
        self.deal_cmd_in_wait_exp_action()
        num_ = 1  # 赋值为1，是因为已经加上了自身
        experiences = []
        while num_ < self.dt_number:
            exp_ = self._other_experience_queue.get()
            experiences.append(exp_)
            num_ += 1
        return self._format_experience(experiences)

    def get_other_agent_experience_target(self):
        num_ = 1  # 赋值为1，是因为已经加上了自身
        experiences = []
        while num_ < self.dt_number:
            exp_ = self._other_experience_queue.get()
            experiences.append(exp_)
            num_ += 1
        for i in experiences:
            self._other_experience_queue.put(i)
        self.deal_cmd_in_wait_running = False

    def _proxy_cmd_with_sending(self, s_, cmd_str, arg, data, blocking=True):
        self.cmd_service.send_cmd(CmdPack.CmdOfStringTable[cmd_str], arg, blocking)
        c, address_ = s_.accept()
        data_p = network.package.DataPackage.DataPack(data)
        c.send(data_p.pack())
        c.close()
        return True

    def broadcast_experience(self, obs_, action_, reward, obs_next):
        data_ = pack_experience(obs_, action_, reward, obs_next)
        s_, _, port_ = get_data_socket(self._address_ip)
        s_.listen()
        self._proxy_cmd_with_sending(s_, "broadcasting state data", port_, data_)
        s_.close()

    def _send_request_for_actions(self, obs_batch):
        data_ = bytes()
        for i in obs_batch:
            data_ = data_ + pack_state(i)
        s_, _, port_ = get_data_socket(self._address_ip)
        s_.listen()
        self._proxy_cmd_with_sending(s_, "request actions", port_, data_)
        s_.close()

    def get_other_agent_action_target(self):
        num_ = 1  # 赋值为1，是因为已经加上了自身
        actions = []
        while num_ < self.dt_number:
            action_ = self._other_actions_queue.get()
            actions.append(action_)
            num_ += 1
        for i in actions:
            self._other_actions_queue.put(i)
        self.deal_cmd_in_wait_running = False

    def get_other_agent_actions(self, obs_batch):
        self._send_request_for_actions(obs_batch)
        threading.Thread(target=self.get_other_agent_action_target).start()
        self.deal_cmd_in_wait_running = True
        self.deal_cmd_in_wait_exp_action()
        num_ = 1  # 赋值为1，是因为已经加上了自身
        actions = []
        while num_ < self.dt_number:
            action_ = self._other_actions_queue.get()
            actions.append(action_)
            num_ += 1
        return self._format_actions(actions, len(obs_batch))

    @classmethod
    def _format_experience(cls, experiences_data):
        exp_ = []
        states = []
        actions = []
        rewards = []
        states_next = []
        for data_, id_ in experiences_data:
            t = unpack_experience(data_)
            exp_.append((id_, t))
        exp_.sort(key=lambda x: x[0])
        exp_ = [i[1] for i in exp_]
        for i in exp_:
            states.append(i[0])
            actions.append(i[1])
            rewards.append(i[2])
            states_next.append(i[3])
        return states, actions, rewards, states_next

    @classmethod
    def _format_actions(cls, actions_data, batch_size):
        actions_ = []
        for data_, id_ in actions_data:
            t = unpack_actions(data_, batch_size)
            actions_.append((id_, t))
        actions_.sort(key=lambda x: x[0])
        actions_ = [i[1] for i in actions_]
        return np.hstack(actions_)

    def get_target_action(self, obs_batch):
        actions_batch = self.ddpg.target_action(obs_batch, self.session)
        return actions_batch


class Cmd5Handle(CmdHandle):

    def before_handle(self, service: CmdService, *args):
        # print("Start dealing Cmd 5.....")
        cmd_: CmdPack = args[0]

    # print(repr(cmd_))

    def handler(self, service: CmdService, *args):
        cmd_: CmdPack = args[0]
        agent_: Agent = args[1]
        port_ = cmd_.arg
        ip_ = service.remote_address[0]
        try:
            data_ = self.receive_data((ip_, port_))
        except ConnectionError as e:
            raise e
        else:
            return data_

    def after_handle(self, service: CmdService, r, *args):
        # print("Finish dealing the Cmd 5 !!")
        return r


class Cmd68Handle(CmdHandle):  # 解包状态，打包动作

    def before_handle(self, service: CmdService, *args):
        cmd_: CmdPack = args[0]
        # print("Start dealing Cmd " + str(cmd_.cmd_code) + ".....")
        # print(repr(cmd_))

    def handler(self, service: CmdService, *args):
        cmd_: CmdPack = args[0]
        agent_: Agent = args[1]
        port_ = cmd_.arg
        ip_ = service.remote_address[0]
        try:
            data_, s_ = self.receive_data_with_no_close_socket((ip_, port_))
        except ConnectionError:
            return False
        else:
            obs_batch = unpack_states(data_)
            actions = agent_.get_target_action(obs_batch)
            action_data = bytes()
            for i in actions:
                action_data = action_data + pack_action(i)
            s_.send(network.package.DataPackage.DataPack(action_data).pack())
            s_.close()
            # 同步每轮中下一状态的动作获取，既保证其它智能体获动作前，该智能体的ddpg目标网络还没有更新
            agent_.request_action_times_the_epoch_ += 1
            return data_

    def after_handle(self, service: CmdService, r, *args):
        cmd_: CmdPack = args[0]
        # print("Finish dealing Cmd " + str(cmd_.cmd_code) + ".....")
        return r


class ReceiveCmdWithIdHandle(CmdHandle):

    def before_handle(self, service: CmdService, *args):
        cmd_: CmdPack = args[0]
        # print("Start dealing Cmd " + str(cmd_.cmd_code) + ".....")
        # print(repr(cmd_))

    def handler(self, service: CmdService, *args):
        cmd_: CmdPack = args[0]
        port_, id_ = cmd_.arg
        ip_ = service.remote_address[0]
        try:
            data_ = self.receive_data((ip_, port_))
        except ConnectionError as e:
            raise e
        else:
            # print("Receive Data: ", len(data_), " Bytes")
            return data_, id_

    def after_handle(self, service: CmdService, r, *args):
        cmd_: CmdPack = args[0]
        # print("Finish dealing Cmd " + str(cmd_.cmd_code) + ".....")
        return r


class Cmd6Handle(CmdHandle):

    def before_handle(self, service: CmdService, *args):
        # print("Start dealing Cmd 6.....")
        cmd_: CmdPack = args[0]
        # print(repr(cmd_))

    def handler(self, service: CmdService, *args):
        cmd_: CmdPack = args[0]
        agent_: Agent = args[1]
        port_ = cmd_.arg
        ip_ = service.remote_address[0]
        data_ = agent_.get_weights()
        try:
            len_ = self.send_data((ip_, port_), data_)
        except ConnectionError:
            return 0
        else:
            return len_

    def after_handle(self, service: CmdService, r, *args):
        # print("Finish dealing the Cmd 6 ,Send data ", r, " bytes")
        return r
