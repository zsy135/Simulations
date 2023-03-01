import tensorflow._api.v2.compat.v1 as tf
# import tensorflow.compat.v1 as tf

tf.compat.v1.disable_v2_behavior()
# tf.disable_v2_behavior()

import config
from train.dataset import DataSet
import matplotlib.pyplot as plt
# from tensorflow.examples.tutorials.mnist import input_data


class MLP:

    def __init__(self, lr, rou):
        self.name = "MLP"
        self.images = tf.placeholder(dtype=tf.float32, shape=[None, 784])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        self.learn_rate = lr
        self.rou = rou

        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.forward()

    def mlp_network(self, ):
        with tf.variable_scope(self.name) as scope:
            x = tf.layers.dense(
                self.images,
                256,
                activation=tf.nn.relu,
                # kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02),
                # bias_initializer=tf.random_normal_initializer(0, 0.01),
              #  name='fc0'
            )

            x = tf.layers.dense(
                x,
                64,
                activation=tf.nn.relu,
                # kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02),
                # bias_initializer=tf.random_normal_initializer(0, 0.01),
                #name='fc1'
            )
            x = tf.layers.dense(
                x,
                10,
                activation=tf.nn.tanh,
                # kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02),
                # bias_initializer=tf.constant_initializer(0, 0.01),
               # name='fc2'
            )
        return x
    def build_assign_graph(self):
        pass

    def forward(self):




        self.optimizer = tf.train.AdamOptimizer(self.learn_rate)
        log_probs = self.mlp_network()
        y = tf.nn.softmax(log_probs)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # 建立优化函数里的正则项的计算图
        self.trainVars = tf.trainable_variables(self.name)
        self.weightPlaceHold = [tf.placeholder(dtype=i.dtype, shape=i.shape) for i in self.trainVars]
        self.differModel = 0
        for i in range(len(self.trainVars)):
            self.differModel += tf.reduce_mean(tf.square(self.trainVars[i] - self.weightPlaceHold[i]))

        # 建立assign 计算图
        self.trainVars = tf.trainable_variables(self.name)
        self.assignWeightPlaceHold = [tf.placeholder(dtype=i.dtype, shape=i.shape) for i in self.trainVars]
        self.assign_opts = []
        for i in range(len(self.trainVars)):
            op = tf.assign(self.trainVars[i], self.assignWeightPlaceHold[i])
            self.assign_opts.append(op)

        #self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.labels * tf.log(y), reduction_indices=[1]))

        self.cross_entropy = tf.reduce_mean(tf.reduce_mean(tf.square(self.labels - y), 1))+ 0.5 * self.rou * self.differModel

        self.mlp_train = self.optimizer.minimize(self.cross_entropy)

    def set_weights(self, sess: tf.Session, weights):
        for i in range(len(self.assign_opts)):
            sess.run(self.assign_opts[i], {self.assignWeightPlaceHold[i]: weights[i]})

    def get_weights(self, sess: tf.Session):
        tmp = []
        for var in self.trainVars:
            tmp.append(sess.run(var))
        return tmp

    def train(self, sess, images, labels, weights):
        feed_dict = self.pack_feed_dict(images, labels, weights)
        sess.run(self.mlp_train, feed_dict=feed_dict)
        return sess.run(self.cross_entropy, feed_dict=feed_dict)

    def get_loss(self, sess, images, labels):
        return sess.run(self.cross_entropy, {self.images: images, self.labels: labels})

    def get_accuracy(self, sess, x, y):
        return sess.run(self.accuracy, {self.images: x, self.y_: y})

    def pack_feed_dict(self, input, target, weights):
        feed_dict = {self.images: input, self.labels: target}
        for i in range(len(weights)):
            feed_dict[self.weightPlaceHold[i]] = weights[i]
        return feed_dict

#
# if __name__ == "__main__":
#     mnist = input_data.read_data_sets("../data/mnist_test", one_hot=True)
#     # train_images_path = "data/train_mnist/train_0_data.npy"
#     # train_labels_path = "data/train_mnist/train_0_label.npy"
#     train_images_path = "../data/train_mnist/train_MnistMerged_data"
#     train_labels_path = "../data/train_mnist/train_MnistMerged_labels"
#     train_dataset = DataSet()
#     train_dataset.load_mnist_images(train_images_path)
#     train_dataset.load_mnist_labels(train_labels_path)
#
#     model = MLP(config.LearnRate, config.Rou)
#     sess = tf.Session()
#
#     sess.run(tf.global_variables_initializer())
#
#     epoth = 500
#     loss_list = []
#     accuracy_list = []
#
#     for i in range(epoth):
#         acc = model.get_accuracy(sess, mnist.test.images, mnist.test.labels)
#         accuracy_list.append(acc)
#         weights = model.get_weights(sess)
#         images, labels = train_dataset.batch(config.BatchSize)
#         loss = 0
#         for j in range(len(images)):
#             loss += model.train(sess, images[j], labels[j], weights)
#
#         loss_list.append(loss / len(images))
#         print("Epoth: ", i, "  Loss: ", loss / len(images), "  Acc: ", acc)
#
#     plt.plot(loss_list, "r-")
#     plt.xlabel("Epoth")
#     plt.ylabel("Loss")
#     plt.figure()
#
#     plt.plot(accuracy_list, "b--")
#     plt.xlabel("Epoth")
#     plt.ylabel("Acc")
#
#     plt.show()
