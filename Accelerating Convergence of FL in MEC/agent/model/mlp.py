import pickle

import tensorflow as tf
from tensorflow.python.layers.layers import Dense
from agent.config import Config


class MLP:
    def __init__(self, input_shape=(None, 784), in_type=tf.float32, output_shape=(None, 10), out_type=tf.float32):
        self.dense1 = Dense(units=Config["mlp_dense1_units"],
                            activation=tf.nn.relu,
                            # kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02),
                            # bias_initializer=tf.random_normal_initializer(0, 0.01),
                            name=Config["mlp_dense1_name"]
                            )
        self.dense2 = Dense(units=Config["mlp_dense2_units"],
                            activation=tf.nn.relu,
                            # kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.02),
                            # bias_initializer=tf.random_normal_initializer(0, 0.01),
                            name=Config["mlp_dense2_name"]
                            )

        self._optimizer = tf.train.AdamOptimizer(1e-4)
        self._inputs = tf.placeholder(dtype=in_type, shape=input_shape)
        self.y_ = tf.placeholder(dtype=out_type, shape=output_shape)
        self._accuracy = None
        self._loss = None
        self._train = None
        self._got_init_w_b = False
        self._init_weights = None
        self.input_len = input_shape[1]
        self.output_len = output_shape[1]
        self.forward()

    def reset(self, session):
        if not self._got_init_w_b:
            self._got_init_w_b = True
            self._init_weights = self.get_weights(session)
        self.set_weights(self._init_weights, session)

    def get_weights(self, session_: tf.Session):
        weights_ = []
        w1 = session_.run(self.dense1.kernel)
        b1 = session_.run(self.dense1.bias)
        w2 = session_.run(self.dense2.kernel)
        b2 = session_.run(self.dense2.bias)
        weights_.append((w1, b1))
        weights_.append((w2, b2))
        return weights_

    def set_weights(self, weights_, session_: tf.Session):
        d1 = weights_[0]
        d2 = weights_[1]
        w1 = d1[0]
        b1 = d1[1]
        w2 = d2[0]
        b2 = d2[1]
        session_.run(tf.assign(self.dense1.kernel, w1))
        session_.run(tf.assign(self.dense1.bias, b1))
        session_.run(tf.assign(self.dense2.kernel, w2))
        session_.run(tf.assign(self.dense2.bias, b2))
        return True

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    def call(self):
        x = self.dense1(self._inputs)
        y = self.dense2(x)
        return y

    def forward(self):
        out = self.call()
        log_probs = tf.nn.softmax(out)
        correct_prediction = tf.equal(tf.argmax(log_probs, 1), tf.argmax(self.y_, 1))
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        self._loss = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(log_probs), reduction_indices=[1]))
        self._train = self._optimizer.minimize(self._loss)

    def train(self, images, labels, sess):
        return sess.run(self._train, {self._inputs: images, self.y_: labels})

    def get_loss(self, images, labels, sess):
        return sess.run(self._loss, {self._inputs: images, self.y_: labels})

    def get_accuracy(self, x, y, sess):
        return sess.run(self._accuracy, {self._inputs: x, self.y_: y})


if __name__ == "__main__":
    session_ = tf.Session()
    mlp = MLP()
    inputs_ = session_.run(tf.random_uniform((10, 784)))
    y_ = session_.run(tf.eye(10, 10))
    init_ = tf.global_variables_initializer()
    session_.run(init_)
    w = mlp.get_weights(session_)
    print(w)
    w_ = mlp.get_weights(session_)
    print(w_)
    b = pickle.dumps(w)
    w_ = pickle.loads(b)
    r_ = None
    for _ in range(10):
        r_ = mlp.train(inputs_, y_, session_)
    loss_ = mlp.get_loss(inputs_, y_, session_)
    accuracy_ = mlp.get_accuracy(inputs_, y_, session_)
    print(r_)
    print(loss_)
    print(accuracy_)
    # print(mlp.get_weights(session_))
