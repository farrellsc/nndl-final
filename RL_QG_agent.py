from keras.models import clone_model, load_model, save_model
from typing import Tuple
import pickle
import os
import random
import logging
import numpy as np
from analyzer import analyzer
import json
from net_struct import *


class RL_QG_agent:
    def __init__(self):
        self.logger = logging.getLogger("agent logger")
        self.__params = json.load(open("./DQN.conf"))

        self.__dynamic_graph = None
        self.__static_graph = None
        self.__iternum = 0
        self.__experience_pool = []
        self.__pool_size = self.__params['RL']['pool_size']

        self.__model_dir = os.path.dirname(os.path.abspath(__file__)) + "/model/"

        self.__init_network(self.__params["NN"]['net'])
        self.analyzer = analyzer()

    def __init_network(self, net_type: str) -> None:
        net_dict = {
            "FCN": FCN,
            "Conv": ConvNet
        }
        self.__dynamic_graph = net_dict[net_type](
            self.__params["NN"]['input_shape'], self.__params["NN"]['output_shape'],
            self.__params["RL"]['lr']
        )
        self.__static_graph = net_dict[net_type](
            self.__params["NN"]['input_shape'], self.__params["NN"]['output_shape'],
            self.__params["RL"]['lr']
        )
        self.__static_graph.model = clone_model(self.__dynamic_graph.model)

    def __calc_dynamic_action(self, state: np.array, enables: np.array, player: int) -> Tuple[np.array, np.array, np.array]:
        state = np.reshape(state, [-1, self.__params['NN']['input_shape'][1]])
        action_probs = self.__dynamic_graph.model.predict(state)
        action_probs = np.reshape(action_probs, [-1, self.__params['NN']['output_shape'][1]])
        if enables == []:
            enables = list(range(action_probs.size))
        for i in range(action_probs.shape[0]):
            action_probs[i, ~np.in1d(range(action_probs.shape[1]), enables[i])] = 0
        action = action_probs.argmax(axis=1)
        action_prob = action_probs[range(action_probs.shape[0]), action]
        return action, action_prob, action_probs

    def __calc_static_action(self, state: np.array, enables: np.array, player: int) -> Tuple[np.array, np.array, np.array]:
        state = np.reshape(state, [-1, self.__params['NN']['input_shape'][1]])
        action_probs = self.__static_graph.model.predict(state)
        action_probs = np.reshape(action_probs, [-1, self.__params['NN']['output_shape'][1]])
        if enables == []:
            enables = list(range(action_probs.size))
        for i in range(action_probs.shape[0]):
            action_probs[i, ~np.in1d(range(action_probs.shape[1]), enables[i])] = 0
        action = action_probs.argmax(axis=1)
        action_prob = action_probs[range(action_probs.shape[0]), action]
        return action, action_prob, action_probs

    def place(self, state: np.array, enables: np.array, player: int, turn: int) -> Tuple[int, list]:
        if random.random() < self.__params['RL']['random_search']/turn:
            action = random.choice(enables)
            return action, []
        else:
            action, action_prob, probs = self.__calc_dynamic_action(state, enables, player)
            # print("current,", action, action_prob, enables)
            return action, probs

    # def learn(self, state_tuple: Tuple[np.array, int, int, np.array, bool, list]) -> float:
    #     # random.shuffle(self.__experience_pool)
    #     # if len(self.__experience_pool) == self.__pool_size: _ = self.__experience_pool.pop(0)
    #     self.__experience_pool.append(state_tuple)
    #     state_in, action, reward, state_out, done, enables = state_tuple
    #
    #     random_index = random.randint(0, len(self.__experience_pool) - 1)
    #     r_state_in, r_action, r_reward, r_state_out, r_done, r_enables = self.__experience_pool[random_index]
    #     r_action, r_action_prob, r_action_probs = self.__calc_static_action(r_state_out, r_enables, 1)
    #     r_value = r_reward if r_done else r_reward + self.__params["RL"]['discount'] * r_action_prob
    #     # print('experience,', r_value, r_action, r_enables)
    #
    #     r_d_action, r_d_action_prob, r_d_action_probs = self.__calc_dynamic_action(r_state_out, r_enables, 1)
    #     r_d_action_probs[r_d_action] = r_value
    #     r_d_action_probs = np.reshape(r_d_action_probs, [1, 64])
    #
    #     # update dynamic graph
    #     state_in = np.reshape(state_in, [1, -1])
    #     history = self.__dynamic_graph.model.fit(state_in, r_d_action_probs, verbose=0)
    #     loss = history.history['loss'][0]
    #
    #     self.__iternum += 1
    #     if self.__iternum % self.__params["RL"]['freeze_interval'] == 0:
    #         self.__static_graph.model = clone_model(self.__dynamic_graph.model)
    #     return loss

    def learn(self, state_tuple: Tuple[np.array, int, int, np.array, bool, np.array]) -> float:
        if len(self.__experience_pool) == self.__pool_size:
            _ = self.__experience_pool.pop(random.randint(0, self.__pool_size - 1))
        self.__experience_pool.append(state_tuple)

        state_in_all = [x[0] for x in self.__experience_pool]
        action_all = [x[1] for x in self.__experience_pool]
        reward_all = [x[2] for x in self.__experience_pool]
        state_out_all = [x[3] for x in self.__experience_pool]
        done_all = [x[4] for x in self.__experience_pool]
        enables_all = [x[5] for x in self.__experience_pool]

        r_action, r_action_prob, r_action_probs = self.__calc_static_action(state_out_all, enables_all, 1)
        r_value = np.array(reward_all) + self.__params['RL']['discount'] * r_action_prob
        r_value[done_all] = np.array(reward_all)[done_all]

        r_d_action, r_d_action_prob, r_d_action_probs = self.__calc_dynamic_action(state_in_all, enables_all, 1)
        r_d_action_probs[range(r_d_action_probs.shape[0]), r_d_action] = r_value

        # update dynamic graph
        state_in_all = np.reshape(state_in_all, [-1, self.__params['NN']['input_shape'][1]])
        history = self.__dynamic_graph.model.fit(state_in_all, r_d_action_probs, verbose=0)
        loss = history.history['loss'][0]

        self.__iternum += 1
        if self.__iternum % self.__params["RL"]['freeze_interval'] == 0:
            self.__static_graph.model = clone_model(self.__dynamic_graph.model)
        return loss

    def agent_load_model(self, fname: str) -> None:
        file_name_no_suffix = fname.split('.')[0]
        self.__dynamic_graph.model = load_model(self.__model_dir + fname)
        self.__static_graph.model = load_model(self.__model_dir + fname)
        self.analyzer = pickle.load(open(self.analyzer.analyzer_dir + file_name_no_suffix + '/' + file_name_no_suffix + '.pkl', 'rb'))

    def agent_save_model(self, fname: str) -> None:
        file_name_no_suffix = fname.split('.')[0]
        save_model(self.__dynamic_graph.model, self.__model_dir + fname)
        self.analyzer.plot_winning_rate(file_name_no_suffix)
        self.analyzer.plot_loss(file_name_no_suffix)
        self.analyzer.plot_reward(file_name_no_suffix)
        self.analyzer.plot_winning_ratio(file_name_no_suffix)
        self.analyzer.plot_turns(file_name_no_suffix)
        pickle.dump(self.analyzer, open(self.analyzer.analyzer_dir + file_name_no_suffix + '/' + file_name_no_suffix + '.pkl', 'wb'))
