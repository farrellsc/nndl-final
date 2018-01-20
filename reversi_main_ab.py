import gym
import random
import numpy as np

from RL_QG_agent import RL_QG_agent
from level1 import alpha_beta as ab

env = gym.make('Reversi8x8-v0')

agent = RL_QG_agent()
max_epochs = 20

w_win =0
b_win =0

for i_episode in range(max_epochs):
    observation = env.reset()
    # agent = RL_QG_agent()
    # observation  是 3 x 8 x 8 的 list,表示当前的棋局，具体定义在 reversi.py 中的 state
    for t in range(100):
        action = [1,2]
        # action  包含 两个整型数字，action[0]表示下棋的位置，action[1] 表示下棋的颜色（黑棋0或者白棋1）

        ################### 黑棋 B ############################### 0表示黑棋
        #  这部分 黑棋 是 用 alpha-beta搜索
        #env.render()  #  打印当前棋局
        enables = env.possible_actions
        if len(enables) == 0:
            action_ = env.board_size**2 + 1
        else:
            # action_ = random.choice(enables)
            action_ = ab.place(observation, enables, 0)#  0 表示黑棋
        action[0] = action_

        action[1] = 0   # 黑棋 B  为 0
        observation, reward, done, info = env.step(action)
        ################### 白棋  W ############################### 1表示白棋
        # env.render()
        enables = env.possible_actions
        # if nothing to do ,select pass
        if len(enables) == 0:
            action_ = env.board_size ** 2 + 1 # pass
        else:
            action_, prob = agent.place(observation, enables, 1, t+1) # 调用自己训练的模型

        action[0] = action_
        action[1] = 1  # 白棋 W 为 1
        observation, reward, done, info = env.step(action)


        if done: # 游戏 结束
            # env.render()
            black_score = len(np.where(env.state[0,:,:]==1)[0])
            if black_score >32:
                print("黑棋赢了！")
                b_win += 1
            else:
                print("白棋赢了！")
                w_win += 1
            print(black_score)
            break
print("黑棋：",b_win,"  白棋 ",w_win)