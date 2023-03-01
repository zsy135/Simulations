import os
import pickle
import random
import threading
import queue
import time

import numpy as np

from network.connect import Connect
import config

#
import tensorflow._api.v2.compat.v1 as tf

# import tensorflow.compat.v1 as tf

tf.compat.v1.disable_v2_behavior()
# tf.disable_v2_behavior()


from model.mlp import MLP
from train.trainTask import TrainTask
from train.dataset import DataSet
from network.packet import *
from collections import defaultdict


class EventType:
    reqWeights = 0
    updateWeights = 1


class Event:
    def __init__(self, eType):
        self.type = eType
        self.doneCondition = threading.Condition()


class ReqWeightEvent(Event):
    def __init__(self):
        super(ReqWeightEvent, self).__init__(EventType.reqWeights)
        self.data = ModelWeiPack()

    def done(self, w, acc, version):
        self.data.weight = w
        self.data.acc = acc
        self.data.version = version
        with self.doneCondition:
            self.doneCondition.notify_all()
        return True


class UpdateWeightEvent(Event):
    def __init__(self, data, client):
        super(UpdateWeightEvent, self).__init__(EventType.updateWeights)
        self.data: ModelWeiPack = data
        self.client = client

    def done(self):
        with self.doneCondition:
            self.doneCondition.notify_all()
        return True


class Client:
    def __init__(self, conn, id_):
        self.connect = conn
        self.id = id_


class Server:
    def __init__(self):
        self.listener = Connect(config.ServerListenIP, config.ServerListenPort)
        self.curTime = 0  # 全局模型的当前版本

        self.alpha = config.Alpha
        self.computeAlphaFun = config.ComputeAlphaFun  # 每个时刻alpha 的计算函数类型

        self.IsFed_Asyn = config.Fed_Asyn

        self.model: MLP = None  # 模型
        self.modelAcc = 0
        self.ModelWeights = None
        self.clients_dict = dict()  # 负责和每个客户端通信的 Connect
        self.updaters = []  # 负责和每个客户端通信的线程

        self.acceptClientWorker = None  # 负责客户接受工作的线程
        self.clientNum = config.ClientNum  # 客户端数目
        self.eventQueue = queue.Queue()  # 事件队列
        self.test_dataset = None  # 测试数据集
        self.printLock = threading.RLock()  # 打印互斥锁

        self.sync_fed_update_queue = queue.Queue()  # 进行同步联邦学习时，更新事件的队列
        self.accuracy = []  # 记录每个时刻模型的准确率
        self.gradients_num = []  # 记录每个时刻 gradient 的次数
        self.per_client_train_times = [0] * self.clientNum
        self.gradients_num_total = 0
        self.session = tf.Session()

        self.train_data_dir = "./data/mnist_client_dataset/"
        self.test_datas_path = "test_data.npy"  # iid 分布
        self.test_labels_path = "test_label.npy"

        self.saverWoker = threading.Thread(target=self.saver)
        self.saveCondition_ = threading.Condition()
        self.blockOwner = [None]
        self.hashFinshedNodesPerRound = defaultdict(list)
        self.prevblockStartTIme = 0

    def notifySaver(self):
        if self.curTime % config.SavaFreq == 0:
            with self.saveCondition_:
                self.saveCondition_.notify_all()

    def saver(self):
        pathResult = "./result/"
        pathModel = "./saved_model/"
        if self.IsFed_Asyn:
            pathResult = pathResult + "asyn/"
            pathModel = pathModel + "asyn/"
        else:
            pathResult = pathResult + "sycn/"
            pathModel = pathModel + "sycn/"
        while True:
            with self.saveCondition_:
                self.saveCondition_.wait()
            with open("./result/roundacc/NodesUpload" + str(self.curTime) + "Round.npy", "wb") as f:
                pickle.dump(self.hashFinshedNodesPerRound, f)
            with open(pathResult + "model_acc_" + str(self.curTime) + "_epoch.npy", "wb") as f:
                pickle.dump(self.accuracy, f)
            with open(pathModel + "model_" + str(self.curTime) + "_epoch.npy", "wb") as f:
                pickle.dump(self.ModelWeights, f)

    def testModel(self):  # 测试模型
        acc = self.model.get_accuracy(self.session, self.test_dataset.images, self.test_dataset.labels)

        n = 10

        images, labels = self.test_dataset.images[:n], self.test_dataset.labels[:n]
        res = self.model.eval(self.session, images, labels)
        slabel = [np.argmax(i) for i in labels]
        print("--------------------------------------------\n")
        f = "     Image {} ---------> {}"
        for i in range(n):
            print(f.format(slabel[i], res[i]))
        print("--------------------------------------------\n")
        return acc

    def prepare(self):  # server 启动前，负责前期的准备工作
        # 设置随机种子

        random.seed(666)
        tf.random.set_random_seed(888)
        np.random.seed(666)

        w = self.bulidModel()
        self.loadTestDataSet()

        self.listener.bind()  # 打开服务器监听端口，监听来自客户端的连接请求
        self.listener.listen()
        self.acceptClients()
        for client in self.clients_dict.values():
            t = threading.Thread(target=self.updater, args=(client,))
            t.start()
            self.updaters.append(t)
        self.saverWoker.start()
        return w

    def bulidModel(self):  # 建立模型
        self.model = MLP(config.LearnRate, config.Rou)
        initer = tf.global_variables_initializer()
        self.session.run(initer)
        return self.getWeights()

    def loadTestDataSet(self):  # 加载测试数据集
        self.test_dataset = DataSet(mnist=True)
        self.test_dataset.load_mnist_images(self.train_data_dir + self.test_datas_path)
        self.test_dataset.load_mnist_labels(self.train_data_dir + self.test_labels_path)

    def acceptClients(self):  # 接受用户连接请求
        num = 0
        print("Start Accept Client:")
        while True:
            client_socket, addr = self.listener.accept()
            print("    Client ", num, " Connected ...")
            cli = Client(Connect(socket_=client_socket, rIp=addr[0], rPort=addr[1]), 0)
            cli.connect.connected = True
            conP: ConnectReqPack = Packet(cli.connect.recieve()).decode()
            cli.id = conP.id
            self.clients_dict[conP.id] = cli
            num += 1
            if num == self.clientNum:
                self.listener.close()
                break
        print("All Client Connect Successfully !!!")

    def start(self):  # 服务器启动入口
        print("Server Start: ..............")
        w = self.prepare()
        self.modelAcc = self.testModel()
        self.broadModelToClients(w, self.modelAcc, self.curTime, None)
        self.prevblockStartTIme = time.time()  # 不用看
        while True:
            if self.curTime == config.EndTime:
                r = input("If continue (yes/no): ")
                r = r.lower()
                if r == "no":
                    os.kill(os.getpid(), 9)
            event = self.eventQueue.get()
            if event.type == EventType.reqWeights:
                self.handleReqWeightsEvent(event)
            elif event.type == EventType.updateWeights:
                self.handleUpdateWeightsEvent(event)
            else:
                raise Exception("Error Event Type !!!")

    def handleReqWeightsEvent(self, event: ReqWeightEvent):  # 负责处理 请求模型参数的 请求事件
        if self.ModelWeights is None:
            self.ModelWeights = self.getWeights()
        event.done(self.ModelWeights, self.modelAcc, self.curTime)

    def getWeights(self):  # 获取模型参数
        # 获取模型参数
        vars_npy = self.model.get_weights(self.session)
        return vars_npy

    def clearEventQueue(self):
        size_ = self.eventQueue.qsize()

        for i in range(size_):
            event = self.eventQueue.get()
            if event.type == EventType.updateWeights:
                self.hashFinshedNodesPerRound[event.client.id].append(
                    (self.curTime, event.data.acc, event.data.version))
                event.done()
            else:
                self.handleReqWeightsEvent(event)

    def handleUpdateWeightsEvent(self, event: UpdateWeightEvent):  # 负责模型的更新
        if self.IsFed_Asyn:
            w = self.doAsynUpdateWeights(event)

            acc = self.testModel()  # 模型更新完成，对模型进行测试

            self.modelAcc = acc
            self.ModelWeights = w
            event.done()  # 模型参数更新完成，让监听更新请求的线程继续接受请求

            self.blockOwner.append(event.client.id)
            self.hashFinshedNodesPerRound[event.client.id].append((self.curTime, event.data.acc, event.data.version))
            self.clearEventQueue()
            self.broadModelToClients(w, acc, self.curTime, event.client.id)

            self.gradients_num_total += config.H_i[event.client.id]
            self.accuracy.append(
                (acc, self.curTime, event.client.id, self.gradients_num_total, time.time() - self.prevblockStartTIme))
            self.prevblockStartTIme = time.time()
            self.notifySaver()
            self.printAcc(acc, event.client.id)

        else:
            self.sync_fed_update_queue.put(event)
            if self.sync_fed_update_queue.qsize() >= self.clientNum:  # 同步联邦学习 ，需要等所有客户端上传参数
                w = self.doSyncUpdateWeights()
                acc = self.testModel()  # 模型更新完成，对模型进行测试
                self.modelAcc = acc
                self.ModelWeights = w
                self.gradients_num_total += config.H_i[event.client.id]
                self.accuracy.append((acc, self.curTime, self.gradients_num_total))
                self.notifySaver()

    def doSyncUpdateWeights(self):  # 同步联邦学习 实际做更新模型参数
        self.curTime += 1
        weights_all_clients = []
        count = 0
        while count < self.clientNum:
            count += 1
            event = self.sync_fed_update_queue.get()
            data = event.data
            task: TrainTask = pickle.loads(data)
            weights_all_clients.append(task.data)
            event.done()  # 模型参数更新完成，让监听更新请求的线程进入监听状态
        weights_new = []
        for i in range(len(weights_all_clients[0])):
            w = []
            for cli in range(self.clientNum):
                w.append(weights_all_clients[cli][i])
            weights_new.append(sum(w) / self.clientNum)
        weights_new = np.array(weights_new)
        # 更新模型
        self.model.set_weights(self.session, weights_new)
        return weights_new

    def doAsynUpdateWeights(self, event: UpdateWeightEvent):  # 异步联邦学习中 实际做更新模型参数

        # 更新模型
        self.curTime += 1
        weights = self.computeNewWeights(event.data.weight, event.data.version)
        self.model.set_weights(self.session, weights)
        return weights

    def computeNewWeights(self, weights, t):  # 计算新的模型参数
        alpha = self.computeAlpha(t)
        weight_old = self.model.get_weights(self.session)
        weights_new = []
        for i in range(len(weights)):
            weights_new.append((1 - alpha) * weight_old[i] + alpha * weights[i])
        return weights_new

    def computeAlpha(self, tao):
        if self.computeAlphaFun == 0:
            return self.constAlpha(tao)
        elif self.computeAlphaFun == 1:
            return self.polynomialAlpha(tao)
        else:
            return self.hingeAlpha(tao)

    def constAlpha(self, tao):
        return 0.8

    def polynomialAlpha(self, tao):
        return 0.4 * (self.curTime - tao + 1) ** (-0.5)

    def hingeAlpha(self, tao):
        b = 4
        if self.curTime - tao <= b:
            return 1
        a = 10
        return 1 / (a * (self.curTime - tao - b) + 1)

    def broadModelToClients(self, weight, acc, version, owner):
        p = ModelBroadCastPack(weight, acc, version, owner)
        data = p.encode()
        for client in self.clients_dict.values():
            client.connect.send(data)

    def updater(self,
                client: Client):  # '更新' 工作者，每个更新者为一个单独的线程，负责接受每个客户的更新模型的请求；；；**** 该函数的功能亦可通过一个线程进行轮询来实现，这里选择多个线程 ****
        while True:
            data = client.connect.recieve()

            p = Packet(data).decode()
            if p.type == PacketType.ModelReq_P:
                reqWeights = ReqWeightEvent()
                self.eventQueue.put(reqWeights)
                with reqWeights.doneCondition:
                    reqWeights.doneCondition.wait()
                client.connect.send(reqWeights.data.encode())

            elif p.type == PacketType.ModelWei_P:
                updateReq = UpdateWeightEvent(p, client)
                self.eventQueue.put(updateReq)
                with updateReq.doneCondition:
                    updateReq.doneCondition.wait()
                # self.printUpdateInfo(client)
            else:
                pass

    def printUpdateInfo(self, client):  # 更新者的打印函数
        self.printLock.acquire()
        print(" ")
        print("********Update Count: ", self.curTime)
        print("    Receive  Weights From client: ", client.id)
        print(" ")
        self.printLock.release()

    def printAcc(self, acc, id_):  # 更新者的打印函数
        self.printLock.acquire()
        print(" ")
        print("******* Model Time: ", self.curTime)
        print("    Model Acc: ", acc)
        print("    Node{} Got An Model Update!".format(id_))
        print(" ")
        self.printLock.release()
