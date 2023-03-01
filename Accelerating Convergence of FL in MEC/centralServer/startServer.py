import pickle
import threading
import time

from network.server.DtServer import DtServer
from config import Config
from network.server.dt import DT
from network.package.CommandPackage import CmdPack


def wait_fed_learn(dt_: DT):
    global finish
    global dt_num
    while True:
        cmd_: CmdPack = dt_.receive_cmd()
        if cmd_.is_upload_weights():
            lock.acquire()
            finish = finish + 1
            weights_ = dt_.execute_cmd(cmd_)
            weights_ = pickle.loads(weights_)
            weights[dt_.id] = weights_
            if finish == dt_num:
                finish = 0
                fed_learn_condition.notifyAll()
            lock.release()
        else:
            dt_.execute_cmd(cmd_)


if __name__ == "__main__":

    server = DtServer(Config["Server_Ip"], Config["Server_Port"])
    t = threading.Thread(target=server.run)
    t.start()
    time.sleep(10)

    lock = threading.RLock()
    finish = 0
    dt_num = server.get_dt_nums()

    fed_learn_condition = threading.Condition()
    weights = {}

    dts_ = server.allocate_n_dt(dt_num)
    for dt in dts_:
        dt_ :DT = dt
        w_ = dt_.get_weights()
        print(w_)
        w_ = pickle.loads(w_)
        print(w_)
        t = threading.Thread(target=wait_fed_learn, args=(dt))
        t.start()

    while True:
        fed_learn_condition.wait()
        for id_, w_ in weights:
            print(w_)
        weights = dict()