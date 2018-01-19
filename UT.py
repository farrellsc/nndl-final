import random
import unittest
import net_struct
import RL_QG_agent
import gym
from utils import *
import numpy as np


class TestSyn(unittest.TestCase):
    def setUp(self):
        self.env = gym.make('Reversi8x8-v0')
        self.env.reset()
        self.agent = RL_QG_agent.RL_QG_agent()
        self.board_latest = None
        self.env.render()
        #self.start_game()

    '''
    def start_game(self):
        action = [1, 2]
        ################### 黑棋 ############################### 0表示黑棋
        #  这部分 黑棋 是随机下棋
        enables = self.env.possible_actions
        if len(enables) == 0:
            action_ = self.env.board_size ** 2 + 1
        else:
            action_ = random.choice(enables)
        action[0] = action_
        action[1] = 0  # 黑棋 为 0
        self.board_latest, reward, done, info = self.env.step(action)
        self.env.render()

    def test_fcn_build(self):
        input_shape = [None, 192]
        output_shape = [None, 64]
        myFCN = net_struct.FCN(input_shape, output_shape)

    def test_agent_setup(self):
        agent = RL_QG_agent.RL_QG_agent()

    def test_place_and_learn(self):
        ################### 白棋 ############################### 1表示白棋
        enables = self.env.possible_actions
        action = [1, 2]
        # if nothing to do ,select pass
        if len(enables) == 0:
            action_ = self.env.board_size ** 2 + 1  # pass
        else:
            action_ = self.agent.place(self.board_latest, enables, 1)  # 调用自己训练的模型

        action[0] = action_
        action[1] = 1  # 白棋 为 1
        observation_out, reward, done, info = self.env.step(action)
        self.env.render()
        self.agent.learn((self.board_latest, action_, reward, observation_out), enables, done)
        self.board_latest = observation_out

    def test_save_load_model(self):
        self.agent.agent_save_model('./model/test.h5')
        new_agent = RL_QG_agent.RL_QG_agent()
        new_agent.agent_load_model('./model/test.h5')
    '''

    def test_set_enable_rankings(self):
        l = list(range(64))
        random.shuffle(l)
        enables = [10,12,14]
        print(l, enables, l[10],l[12],l[14])
        gm = gameInfo()
        gm.set_enable_rankings(enables, l)
        print(gm.enable_ranking)
