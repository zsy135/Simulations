import os
import threading
import time

import psutil
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from agent.model.replay_buffer import ReplayBuffer
from agent.config import Config
from agent.model.ddpg import DDPG
from agent.model.env import Env
from agent.model.mlp import MLP
import csv


def build_ddpg_update_graph(tau=0.99):
    online_action_variables = [i for i in tf.trainable_variables() if "online_action" in i.name]
    target_action_variables = [i for i in tf.trainable_variables() if "target_action" in i.name]
    online_critic_variables = [i for i in tf.trainable_variables() if "online_critic" in i.name]
    target_critic_variables = [i for i in tf.trainable_variables() if "target_critic" in i.name]
    action_vars = zip(online_action_variables, target_action_variables)
    critic_vars = zip(online_critic_variables, target_critic_variables)
    target_online_assign_actor = []
    target_online_assign_critic = []
    for online_, target_ in action_vars:
        op = tf.assign(target_, (1 - tau) * online_ + tau * target_)
        target_online_assign_actor.append(op)
    for online_, target_ in critic_vars:
        op = tf.assign(target_, (1 - tau) * online_ + tau * target_)
        target_online_assign_critic.append(op)
    return target_online_assign_actor, target_online_assign_critic


class Train:
    def __init__(self, agent_p):
        self._agent = agent_p
        self._session: tf.Session = self._agent.session
        self._my_replay_buffer = ReplayBuffer(Config["replay_buffer_size"])
        self._count_experience = 0
        self.sample_number = 32
        self._mlp: MLP = self._agent.mlp
        self._ddpg: DDPG = self._agent.ddpg
        self._env = Env(self._mlp, self._session)
        self._fl_frequency = 10
        self._train_mlp_epochs = 100
        self._start_ddpg_train = 1000
        self._reward = []
        self._accuracy = []
        self._resources = []
        self._actions = []
        self._loss_action = []
        self._loss_critic = []
        self._epoch = 0
        # Energy Measure
        self.CPU_FREQ = 1.4  # GHZ
        self.lamada = 0.1
        self.cpu_load_measure_time = 0.1
        self.process = psutil.Process(os.getpid())
        self.wait_record_cpu_percent_cond = threading.Condition()
        self.is_set_recod_stoped = False
        self.event_record_cpu_p_finished = threading.Event()
        self.target_online_assign_actor, self.target_online_assign_critic = build_ddpg_update_graph()
        self.cpu_percent_record_data = None
        self.all_cpu_percent_records_data = []

        self.acc_file = open("./accuracy.txt", "w+")
        self.res_file = open("./resource.txt", "w+")
        self.rew_file = open("./reward.txt", "w+")
        self.act_file = open("./actions.txt", "w+")

        self.wait_duration_file = open("./duration_wait.txt", "w+")

        self.CPU_percent_writer = open("./CPU_Percent.txt", "w+")

        self.acc_writer = csv.writer(self.acc_file)
        self.res_writer = csv.writer(self.res_file)
        self.rew_writer = csv.writer(self.rew_file)
        self.act_writer = csv.writer(self.act_file)
        self.wait_duration_writer = csv.writer(self.wait_duration_file)
        self.start_worker()

    def flush_file(self):
        self.acc_file.flush()
        self.res_file.flush()
        self.rew_file.flush()
        self.act_file.flush()
        self.wait_duration_file.flush()


    def start_worker(self):
        t = threading.Thread(target=self.get_cpu_percent)
        t.start()

    def stop_worker(self):
        self.is_set_recod_stoped = True
        self.event_record_cpu_p_finished.clear()
        with self.wait_record_cpu_percent_cond:
            self.wait_record_cpu_percent_cond.notifyAll()

    def update_target(self, session_: tf.Session):
        for op in self.target_online_assign_actor:
            session_.run(op)
        for op in self.target_online_assign_critic:
            session_.run(op)
        return True

    def get_cpu_percent(self):
        while not self.is_set_recod_stoped:
            with self.wait_record_cpu_percent_cond:
                self.wait_record_cpu_percent_cond.wait()
            while self.event_record_cpu_p_finished.is_set():
                percent_ = self.process.cpu_percent(self.cpu_load_measure_time)
                self.CPU_percent_writer.write(str(percent_) + ", ")
                self.cpu_percent_record_data.append(percent_)

    def compute_enegy_cost(self):
        energy = []
        for i in self.all_cpu_percent_records_data:
            e_ = 0
            for j in i:
                e_ += self.cpu_load_measure_time * self.lamada * ((j/100 * self.CPU_FREQ) ** 3)
            energy.append(e_)
        # print(energy)
        return energy

    def _train_ddpg(self):
        obs_batch, actions_batch, reward_batch, obs_next_batch, done_batch = self._my_replay_buffer.sample(
            self.sample_number)  # 32改64 不行 改为16 不行改为8
        obs_batch = obs_batch[:, 0, :]
        reward_batch = reward_batch[:, 0, :]
        obs_next_batch = obs_next_batch[:, 0, :]

        other_actions_batch = np.hstack([actions_batch[:, i, :] for i in range(1, self._agent.dt_number)])

        other_actions_next_batch = self._agent.get_other_agent_actions(obs_next_batch)

        next_actions_batch = self._ddpg.action(obs_next_batch, self._session)

        target = reward_batch.reshape(-1, 1) + 0.99 * self._ddpg.target_q(obs_next_batch,
                                                                          next_actions_batch, other_actions_next_batch,
                                                                          self._session)

        loss_action = self._ddpg.train_action(obs_batch, other_actions_batch, self._session)
        loss_critic = self._ddpg.train_critic(obs_batch, actions_batch[:, 0, :], other_actions_batch, target,
                                              self._session)
        # print("Action got!!")
        # times = 0
        while self._agent.request_action_times_the_epoch_ < self._agent.dt_number:
            # if times < 10:
            #     print("In deal action req!!  ", self._agent.request_action_times_the_epoch_)
            # times += 1
            self._agent.deal_action_request_cmd()
        self._agent.request_action_times_the_epoch_ = 1
        self.update_target(self._session)
        return loss_action, loss_critic

    def interactive_env(self, obs_):
        env: Env = self._env
        obs_ = obs_.reshape((-1, 4))
        action = self._ddpg.action(obs_, self._session)
        action = action.flatten()
        epochs = self._ddpg.action_to_epochs(action)
        self._actions.append(epochs)
        obs_next, reward, pos_reward, neg_reward = env.step(epochs, obs_)

        obs_flat = obs_.flatten()
        self._agent.broadcast_experience(obs_flat, action, reward, obs_next)

        other_obs, other_actions, other_rewards, other_obs_next = self._agent.get_other_agent_experience()

        self._my_replay_buffer.add(np.vstack([obs_flat, other_obs]),
                                   np.vstack([action, other_actions]),
                                   np.vstack([reward, other_rewards]),
                                   np.vstack([obs_next, other_obs_next]), False)

        self._count_experience += 1
        return obs_next, reward, pos_reward, neg_reward

    def finish_train(self):
        self.acc_file.close()
        self.res_file.close()
        self.rew_file.close()
        self.act_file.close()
        self.wait_duration_file.close()
        self.stop_worker()
        time.sleep(2)
        self.CPU_percent_writer.close()
        self.plot()

    def plot(self):
        y = self.compute_enegy_cost()
        x = list(range(len(y)))
        plt.plot(x, y)
        plt.show()

    def clear_recoder(self):
        self._loss_action.clear()
        self._loss_critic.clear()
        self._actions.clear()
        self._accuracy.clear()
        self._reward.clear()
        self._resources.clear()

    def step(self):
        self.clear_recoder()
        env: Env = self._env
        env.reset()
        obs_ = env.get_state()
        self._resources.append(obs_[1])
        self._accuracy.append(obs_[2])
        self._epoch += 1
        obs_ = obs_.reshape((-1, 4))
        # print('init_state:{}'.format(obs_))
        print('Epoch:{}'.format(self._epoch))
        reward_sum = 0
        self.CPU_percent_writer.write("\n")
        self.cpu_percent_record_data = []
        duration_wait = []
        # time_cost_mlp_train = 0
        for i in range(self._train_mlp_epochs):
            print(i)

            self.event_record_cpu_p_finished.set()
            with self.wait_record_cpu_percent_cond:
                self.wait_record_cpu_percent_cond.notifyAll()
            # start_fd_learn_time1 = time.time()
            obs_next, reward, os_reward, neg_reward = self.interactive_env(obs_)
            self.event_record_cpu_p_finished.clear()
            # start_fd_learn_time1 = time.time() - start_fd_learn_time1
            # time_cost_mlp_train += start_fd_learn_time1
            # print(str(start_fd_learn_time1))
            self._resources.append(obs_next[1])
            self._accuracy.append(obs_next[2])
            reward_sum = reward_sum + reward
            if self._count_experience >= self._start_ddpg_train:
                loss_a, loss_b = self._train_ddpg()
                self._loss_action.append(loss_a)
                self._loss_critic.append(loss_b)
            if (i + 1) % self._fl_frequency == 0 :
                start_wait = time.time()
                self.event_record_cpu_p_finished.set()
                with self.wait_record_cpu_percent_cond:
                    self.wait_record_cpu_percent_cond.notifyAll()
                # start_fd_learn_time2 = time.time()
                self._agent.fed_learn_once()
                self.event_record_cpu_p_finished.clear()
                # start_fd_learn_time2 = time.time() - start_fd_learn_time2
                # time_cost_mlp_train += start_fd_learn_time2
                # print(str(start_fd_learn_time2))
                duration_wait.append(time.time() - start_wait)
                # self._env.fed_learn_once()
                # print(env.get_state().reshape((-1, 4)))
            obs_ = obs_next
        self.all_cpu_percent_records_data.append(self.cpu_percent_record_data)
        self.CPU_percent_writer.flush()
        self.wait_duration_writer.writerow(duration_wait)
        self._reward.append(reward_sum)
        # if self._loss_criti
        #     print("Loss Action: ", np.mean(self._loss_action))
        #     print("Actions: ", self._actions)
        #     print("Loss Critic: ", np.mean(self._loss_critic))
        self.acc_writer.writerow(self._accuracy)
        self.res_writer.writerow(self._resources)
        self.rew_writer.writerow(self._reward)
        self.act_writer.writerow(self._actions)
        if self._epoch % 1== 0:
            self.flush_file()
        # print("Resource: ", self._resources)
        # print("Reward: ", self._reward)
        # print("Accuracy: ", self._accuracy)
        # print("Actions: ", self._actions)
