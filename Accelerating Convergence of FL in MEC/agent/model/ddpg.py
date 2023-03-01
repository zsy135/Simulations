import numpy as np
import tensorflow as tf
from agent.config import Config


class DDPG:
    def __init__(self, layer_norm=True):
        self._action_space = np.arange(0, 10)
        self._nb_state = 4
        self._nb_actions = len(self._action_space)
        self._nb_other_action = (Config["number_of_agent"] - 1) * self._nb_actions
        self._layer_norm = layer_norm
        self._state_input = tf.placeholder(shape=[None, self._nb_state], dtype=tf.float32)
        self._action_input = tf.placeholder(shape=[None, self._nb_actions], dtype=tf.float32)
        self._other_action_input = tf.placeholder(shape=[None, self._nb_other_action], dtype=tf.float32)
        self._reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self._trainable_var_action = []
        self._trainable_var_critic = []

        self.noise_rate = 0.4

        # 1e-3 1e-3  学习率从 1e-3 改为 1e-4和 1e-3
        self._actor_optimizer = tf.train.AdamOptimizer(1e-5)
        self._critic_optimizer = tf.train.AdamOptimizer(1e-4)

        self._online_action = None
        self._target_action = None

        self._loss_action = None
        self._train_action = None

        self._online_critic = None
        self._target_critic = None

        self._loss_critic = None
        self._train_critic = None

        self._target_Q = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self.forward()

    def action_network(self, name):
        with tf.variable_scope(name, reuse=False):
            x = tf.layers.dense(self._state_input, 128)
            if self._layer_norm:
                x = tf.layers.BatchNormalization(center=True, scale=True)(x)
            x = tf.nn.relu(x)
            x = tf.layers.dense(x, 64)
            if self._layer_norm:
                x = tf.layers.BatchNormalization(center=True, scale=True)(x)
            x = tf.nn.relu(x)
            x = tf.layers.dense(x, self._nb_actions,
                                kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        # y = tf.nn.softmax(x)
        # y = gumbel_softmax(x, 1)
        return x

    def critic_network(self, name):
        with tf.variable_scope(name, reuse=False):
            x = tf.layers.dense(self._state_input, 128)
            if self._layer_norm:
                x = tf.layers.BatchNormalization(center=True, scale=True)(x)
            x = tf.nn.relu(x)
            x = tf.concat([x, self._online_action, self._other_action_input], axis=-1)
            x = tf.layers.dense(x, 64)
            if self._layer_norm:
                x = tf.layers.BatchNormalization(center=True, scale=True)(x)
            x = tf.nn.tanh(x)
            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
        return x

    def forward(self):

        self._online_action = self.action_network("online_action")
        self._target_action = self.action_network("target_action")
        self._online_critic = self.critic_network("online_critic")
        self._target_critic = self.critic_network("target_critic")

        self._trainable_var_action = [i for i in tf.trainable_variables() if "online_action" in i.name]
        self._trainable_var_critic = [i for i in tf.trainable_variables() if "online_critic" in i.name]

        # 最大化Q值
        self._loss_action = -tf.reduce_mean(self._online_critic)
        self._train_action = self._actor_optimizer.minimize(self._loss_action, var_list=self._trainable_var_action)

        self._loss_critic = tf.reduce_mean(tf.square(self._target_Q - self._online_critic))
        self._train_critic = self._critic_optimizer.minimize(self._loss_critic)

    def train_action(self, state, other_action, sess):
        return sess.run(self._train_action, {self._state_input: state, self._other_action_input: other_action})

    def train_critic(self, state, action, other_action, target, sess):
        return sess.run(self._train_critic,
                        {self._state_input: state, self._action_input: action, self._other_action_input: other_action,
                         self._target_Q: target})

    def action(self, state, sess):
        actions = sess.run(self._online_action, {self._state_input: state})
        actions = actions + np.random.randn(10) * self.noise_rate
        return actions

    def action_to_epochs(self, actions):
        index_ = np.argmax(actions, axis=-1)
        return self._action_space[index_]

    def target_action(self, state, sess):
        actions = sess.run(self._target_action, {self._state_input: state})
        return actions

    def target_q(self, state, action, other_action, sess):
        return sess.run(self._target_critic,
                        {self._state_input: state, self._action_input: action, self._other_action_input: other_action})


if __name__ == "__main__":
    ddpg = DDPG()
    print(" ")
