import random
import threading
import time

from network.connect import Connect
import config
import queue
from train.trainTask import TrainTask
import pickle


import tensorflow._api.v2.compat.v1 as tf
# import tensorflow.compat.v1 as tf

tf.compat.v1.disable_v2_behavior()


from model.mlp import MLP
from train.dataset import DataSet
# from tensorflow.examples.tutorials.mnist import input_data
from threading import Event
import hashlib
import time
from network.packet import *

max_nonce = 2 ** 32  # 4 billion


class State:
    Idle = 0
    Running = 1


class Worker:
    def __init__(self, id_, malicious=False):
        self.id = id_  # worker 的id
        self.state = State.Idle  # worker工作状态
        self.connect = None  # 与服务器的连接
        self.stop = True  # 是否运行工作者

        self.H_i = config.H_i[self.id]
        self.queueTrainTask = queue.Queue()

        self.recieverThread = None

        self.rou = config.Rou
        self.learn_rate = config.LearnRate

        self.session = tf.Session()

        self.model: MLP = None # m模型

        self.batchSize = config.BatchSize

        self.weights = None
        self.dataset = None
        self.train_data_dir = "./data/mnist_client_dataset/"

        if malicious:
            self.train_datas_path = "train_" + str(self.id) + "_malicious_data.npy"  # iid 分布
            self.train_labels_path = "train_" + str(self.id) + "_malicious_label.npy"
        else:
            self.train_datas_path = "train_" + str(self.id) + "_data.npy"  # iid 分布
            self.train_labels_path = "train_" + str(self.id) + "_label.npy"

        self.test_datas_path = "test_data.npy"  # iid 分布
        self.test_labels_path = "test_label.npy"

        self.updataRound = 0
        # self.train_datas_path = "train_" + str(self.id) + "_data" + "_nonid" + ".npy"  # non iid 分布
        # self.train_labels_path = "train_" + str(self.id) + "_label" + "_nonid" + ".npy"
        self.loss_all_time = {}

        self.newestVersion = 0
        self.newestVersionAcc = 0
        self.newVersionEvent = threading.Event()

        self.blockOwner = []

        self.olderVersion = -1


    def isRun(self):
        return self.state == State.Running

    def preTrain(self, task):
        # 取出最新的模型
        self.updataRound += 1
        while self.queueTrainTask.qsize() > 0:
            tmp = self.queueTrainTask.get()
            if tmp.version > task.version:
                task = tmp
        self.updateModel(task.weight)
        return task

    def train(self, task):
        loss = []
        print("Start Training: ")
        for i in range(self.H_i):
            images_batch, labels_batch = self.dataset.batch(self.batchSize)
            l = []
            for j in range(len(images_batch)):
                tmp = self.model.train(self.session, images_batch[j], labels_batch[j], task.weight)
                l.append(tmp)
            # images_train, labels_train = self.dataset.sample(100)
            # tmp = self.model.train(self.session, images_train, labels_train, task.data)
            # tmp = self.model.train(self.session, self.dataset.images, self.dataset.labels, task.data)
            # l.append(tmp)
            loss_epoth = sum(l) / len(l)
            loss.append(loss_epoth)
            print("Model Version: ", task.version, " Epoth: ", i, " Loss: ", loss_epoth)
        self.loss_all_time[task.version] = loss

    def testModel(self):
        """
        测试模型精度
        :return: 精度
        """
        acc = self.model.get_accuracy(self.session, self.test_dataset.images, self.test_dataset.labels)
        return acc

    def loadTrainDataSet(self):
        self.dataset = DataSet(mnist=True)
        self.dataset.load_mnist_images(self.train_data_dir + self.train_datas_path)
        self.dataset.load_mnist_labels(self.train_data_dir + self.train_labels_path)
        self.test_dataset = DataSet(mnist=True)
        self.test_dataset.load_mnist_images(self.train_data_dir + self.test_datas_path)
        self.test_dataset.load_mnist_labels(self.train_data_dir + self.test_labels_path)

    def buildModel(self):
        self.model = MLP(self.learn_rate, self.rou)
        initer = tf.global_variables_initializer()
        self.session.run(initer)

    def getModelWeight(self):
        return self.model.get_weights(self.session)

    def updateModel(self, weights):
        self.weights = weights
        return self.model.set_weights(self.session, weights)

    def toIdle(self):
        self.state = State.Idle
        print("Model Quit Train!!!")

    def clearTrainTaskQuene(self):
        for i in range(self.queueTrainTask.qsize()):
            self.queueTrainTask.get()

    def restartTrain(self):  # 重新加入训练
        self.toRunning()
        print("Model Restart Train!!!")
        self.clearTrainTaskQuene()
        while True:
            tmp = WeiReqPack()
            self.connect.send(tmp.encode())

    def quitTrain(self):
        if self.updataRound % 10 == 0:
            if random.random() > 0.9:  # 退出
                while True:
                    self.toIdle()
                    time.sleep(10)
                    if random.random() <= 0.95:
                        break
                return True
            else:
                return False
        else:
            return False

    def toRunning(self):
        self.state = State.Running

    def prepare(self):
        self.loadTrainDataSet()
        self.buildModel()
        self.connect = Connect()
        self.connect.connect(config.ServerListenIP, config.ServerListenPort)
        self.connect.send(ConnectReqPack(id_=self.id).encode())
        self.recieverThread = threading.Thread(target=self.recieveModel)
        self.recieverThread.start()

    def recieveModel(self):
        while True:
            data = self.connect.recieve()
            pack = Packet(data).decode()
            if pack.type == PacketType.ModelBroadCast_P and pack.version > self.newestVersion:
                self.newestVersion = pack.version
                self.newVersionEvent.set()
                self.newestVersionAcc = pack.acc
                self.blockOwner.append(pack.owner)
            self.queueTrainTask.put(pack)

    def computeBlock(self, modelAcc, task):
        found = False
        breakNums = 0
        while breakNums <= 2 and not found:
            difficulty = (236 + (modelAcc - self.newestVersionAcc) * 150)
            target = 2 ** difficulty
            print("My Difficuty: ", difficulty)
            for nonce in range(max_nonce):
                if self.newVersionEvent.isSet():
                    breakNums += 1
                    self.newVersionEvent.clear()
                    break
                hash_result = hashlib.sha256(
                    (str(task.version) + str(self.newestVersion) + str(nonce)).encode("utf-8")).hexdigest()
                if int(hash_result, 16) < target:
                    found = True
                    break
        return found

    def afterTrain(self, task):
        modelAcc = self.testModel()
        print("\n**** Model Acc: {} \n".format(modelAcc))

        # found = self.computeBlock(modelAcc, task)
        # if found:
        if True:
            weights = self.getModelWeight()
            pack = ModelWeiPack(weights, modelAcc, task.version)
            self.connect.send(pack.encode())
            # self.olderVersion = self.newestVersion

    def waitForNewer(self):
        while True:
            task = self.queueTrainTask.get()
            if task.version > self.olderVersion:
                return task


    def start(self):
        self.prepare()
        self.stop = False
        self.toRunning()
        while not self.stop:
            task = self.waitForNewer()
            if not self.quitTrain():
                task = self.preTrain(task)
                self.train(task)
                self.afterTrain(task)
            else:
                self.restartTrain()
