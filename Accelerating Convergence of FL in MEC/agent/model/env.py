import pickle

from agent.model.mlp import MLP
import numpy as np
import tensorflow as tf


class Mnist:
    class Data:
        def __init__(self):
            self.images = None
            self.labels = None

    def __init__(self, path):
        data = tf.keras.datasets.mnist.load_data(path)
        # self._train_data_count = len(data[0][1])
        self._test_data_count = len(data[1][1])
        self.test = Mnist.Data()
        # self.train = Mnist.Data()
        y2_ = self._format(data)
        # self.data = ((data[0][0].reshape(10000, -1), y1_), (data[1][0].reshape(10000, -1), y2_))
        # self.train.images = data[0][0].reshape(10000, -1)
        # self.train.labels = y1_
        self.test.images = data[1][0].reshape(10000, -1)
        self.test.labels = y2_

    # def dump(self, path):
    #     file_ = open(path, "wb")
    #     pickle.dump(self.data, file_)
    #     file_.close()

    def _format(self, data):
        # y1_ = np.zeros((self._train_data_count, 10))
        y2_ = np.zeros((self._test_data_count, 10))
        # for i in range(self._train_data_count):
        #     y1_[i][data[0][1][i]] = 1
        for i in range(self._test_data_count):
            y2_[i][data[1][1][i]] = 1
        # return y1_, y2_
        return y2_


class Env:
    def __init__(self, mlp_, sess_):
        self._session: tf.Session = sess_
        self._mlp: MLP = mlp_
        self._RES = 12  # 计算reward时资源的权重
        self._resource = 1.0
        self._accuracy = 0
        self._loss = 0
        self._done = False

        # self._comp_resource = 0.0008
        self._comp_resource = 0.0023
        self._comm_resource = 0.016
        self._batch_size = 80
        self._acc_hold = 0.880

        self._input_len = self._mlp.input_len
        self._out_len = self._mlp.output_len
        self._train_data = np.load("./train_data.npy").reshape((-1, self._batch_size, self._input_len))
        self._train_labels = np.load("./train_label.npy").reshape((-1, self._batch_size, self._out_len))
        self._mnist = Mnist("./mnist.npz")
        self._init()

    def _init(self):
        self._resource = 1.0
        self._loss = self.get_loss()
        self._accuracy = self.get_accuracy()

    def get_loss(self):
        _loss = []
        for images, labels in zip(self._train_data, self._train_labels):
            loss = self._mlp.get_loss(images, labels, self._session)
            _loss.append(loss)
        r = np.mean(_loss)
        return r

    def get_accuracy(self):
        r = self._mlp.get_accuracy(self._mnist.test.images, self._mnist.test.labels, self._session)
        return r

    def reset(self):
        self._mlp.reset(self._session)
        self._init()

    def get_state(self):
        return np.array([self._loss, self._resource, self._accuracy, self._done])

    def step(self, epochs, obs_):
        obs_ = obs_.flatten()
        if obs_[1] - self._comp_resource * epochs < 0 or obs_[1] - self._comm_resource < 0:
            self._done = True
        for _ in range(epochs):
            for images, labels in zip(self._train_data, self._train_labels):
                self._mlp.train(images, labels, self._session)

        self._loss = self.get_loss()
        self._resource = self._resource - self._comp_resource * epochs
        self._accuracy = self.get_accuracy()
        neg_reward = self._comp_resource * epochs
        pos_reward = self._accuracy - self._acc_hold
        if pos_reward < 0:
            pass
        else:
            pos_reward *= 20
        neg_reward = (neg_reward / np.clip(obs_[1], 0.04, 1)) * self._RES
        reward = pos_reward - neg_reward
        return self.get_state(), np.array([reward]), pos_reward, neg_reward


if __name__ == "__main__":
    mnist = Mnist("./mnist.npz")
