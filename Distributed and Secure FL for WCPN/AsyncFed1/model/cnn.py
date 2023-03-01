import tensorflow._api.v2.compat.v1 as tf
# import tensorflow.compat.v1 as tf

tf.compat.v1.disable_v2_behavior()
# tf.disable_v2_behavior()

import config
from train.dataset import DataSet
import matplotlib.pyplot as plt


class CNN:

    def __init__(self, lr, rou):
        self.input = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3))
        self.target = tf.placeholder(dtype=tf.float32, shape=(None, 10))
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, 10))
        self.name = "CNN"
        self.learn_rate = lr
        self.rou = rou
        self.forward()

    def cnn_network(self, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()
            x = self.input
            # Layer Conv 1
            x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu)
            x = tf.layers.batch_normalization(x)
            # Layer Conv 2
            x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu)
            x = tf.layers.batch_normalization(x)
            # Pool Layer
            x = tf.layers.max_pooling2d(x, 2, 2)
            # Dropout1
            x = tf.layers.dropout(x, 0.25)
            # Layer Conv 3
            x = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu)
            x = tf.layers.batch_normalization(x)
            # Layer Conv 4
            x = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu)
            x = tf.layers.batch_normalization(x)
            # Pool Layer
            x = tf.layers.max_pooling2d(x, 2, 2)
            x = tf.layers.flatten(x)
            # Dense 1
            x = tf.layers.dense(x, 128, tf.nn.relu)
            # Drop Layer 3
            x = tf.layers.dropout(x, 0.25)
            # Dense 3
            x = tf.layers.dense(x, 10, tf.nn.tanh)
            x = tf.nn.softmax(x)
            return x

    # def cnn_network(self, reuse=False):
    #     with tf.variable_scope(self.name) as scope:
    #         if reuse:
    #             scope.reuse_variables()
    #         x = self.input
    #         # Layer Conv 1
    #         x = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu)
    #         x = tf.layers.batch_normalization(x)
    #         # Layer Conv 2
    #         x = tf.layers.conv2d(x, 64, 3, activation=tf.nn.relu)
    #         x = tf.layers.batch_normalization(x)
    #         # Pool Layer
    #         x = tf.layers.max_pooling2d(x, 2, 2)
    #         # Dropout1
    #         x = tf.layers.dropout(x, 0.25)
    #         # Layer Conv 3
    #         x = tf.layers.conv2d(x, 128, 3, activation=tf.nn.relu)
    #         x = tf.layers.batch_normalization(x)
    #         # Layer Conv 4
    #         x = tf.layers.conv2d(x, 128, 3, activation=tf.nn.relu)
    #         x = tf.layers.batch_normalization(x)
    #         # Pool Layer
    #         x = tf.layers.max_pooling2d(x, 2, 2)
    #         x = tf.layers.flatten(x)
    #         # Dense 1
    #         x = tf.layers.dense(x, 256, tf.nn.relu)
    #         # Drop Layer 3
    #         x = tf.layers.dropout(x, 0.25)
    #         # Dense 3
    #         x = tf.layers.dense(x, 10, tf.nn.relu)
    #         x = tf.nn.softmax(x)
    #         return x

    def forward(self):
        self.output = self.cnn_network()

        self.trainVars = tf.trainable_variables(self.name)
        self.weightPlaceHold = [tf.placeholder(dtype=i.dtype, shape=i.shape) for i in self.trainVars]
        self.differModel = 0
        for i in range(len(self.trainVars)):
            self.differModel += tf.reduce_mean(tf.square(self.trainVars[i] - self.weightPlaceHold[i]))

        # 建立assign 计算图

        self.assignWeightPlaceHold = [tf.placeholder(dtype=i.dtype, shape=i.shape) for i in self.trainVars]
        self.assign_opts = []
        for i in range(len(self.trainVars)):
            op = tf.assign(self.trainVars[i], self.assignWeightPlaceHold[i])
            self.assign_opts.append(op)

        self.differModel = self.differModel / len(self.trainVars)

        self.optimizer = tf.train.AdamOptimizer(1e-2)

        self.loss = tf.reduce_mean(
            tf.reduce_mean(tf.square(self.output - self.target), 1))  # + 0.0 * config.Rou * self.differModel
        # self.loss = tf.reduce_mean(-tf.reduce_sum(self.target * tf.log(self.output), reduction_indices=[1]))

        self.train = self.optimizer.minimize(self.loss)

        correct_prediction = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def set_weights(self, sess: tf.Session, weights):
        for i in range(len(self.assign_opts)):
            sess.run(self.assign_opts[i], {self.assignWeightPlaceHold[i]: weights[i]})

    def get_weights(self, sess: tf.Session):
        tmp = []
        for var in self.trainVars:
            tmp.append(sess.run(var))
        return tmp

    def train_network(self, sess: tf.Session, input, target, weights):
        feed_dict = self.pack_feed_dict(input, target, weights)
        sess.run(self.train, feed_dict=feed_dict)
        return sess.run(self.loss, feed_dict=feed_dict)

    def predict(self, sess: tf.Session, input):
        return sess.run(self.output, {self.input: input})

    def get_accuracy(self, sess: tf.Session, input, y):
        return sess.run(self.accuracy, {self.input: input, self.y: y})

    def pack_feed_dict(self, input, target, weights):
        feed_dict = {self.input: input, self.target: target}
        for i in range(len(weights)):
            feed_dict[self.weightPlaceHold[i]] = weights[i]
        return feed_dict


# mnist_test model


if __name__ == "__main__":
    train_dataset_path = "../data/train_cifar/train_1.npy"
    test_dataset_path = "../data/train_cifar/test.npy"
    train_dataset = DataSet(train_dataset_path)
    test_dataset = DataSet(test_dataset_path)

    model = CNN()
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    epoth = 500
    loss_list = []
    accuracy_list = []
    weights = model.get_weights(sess)
    for i in range(epoth):
        acc = model.get_accuracy(sess, test_dataset.images, test_dataset.labels)
        accuracy_list.append(acc)

        images, labels = train_dataset.batch(config.BatchSize)
        loss = 0
        for j in range(len(images)):
            loss += model.train_network(sess, images[j], labels[j], weights)

        loss_list.append(loss / len(images))
        print("Epoth: ", i, "  Loss: ", loss / len(images), "  Acc: ", acc)

    plt.plot(loss_list, "r-")
    plt.plot(accuracy_list, "b--")
    plt.xlabel("Epoth")
    plt.ylabel("Loss")
    plt.show()
