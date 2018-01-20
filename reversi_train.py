import os
import gym
import random
from tqdm import tqdm
from utils import *
import numpy as np
from RL_QG_agent import RL_QG_agent
import json

params = json.load(open('./general.conf'))
agent_name = params['agent_name'] + '_' + params['hand'] + '.h5'
white_win = 0
black_win = 0
inval = 0
params['new_agent'] = False if os.path.isfile('./model/' + agent_name) else True

env = gym.make('Reversi8x8-v0')
env.reset()
agent = RL_QG_agent()

if not params['new_agent']:
    agent.agent_load_model(agent_name)
    print('Updating existing agent...')
else:
    os.mkdir('./analyze/' + params['agent_name'] + '_' + params['hand'])
    print('training new agent...')

for i_episode in tqdm(range(params["max_epochs"])):
    observation = env.reset()
    # observation  是 3 x 8 x 8 的 list,表示当前的棋局，具体定义在 reversi.py 中的 state
    current_game_info = gameInfo()
    current_reward = None

    for t in range(100):
        action = [1, 2]     # [pos, color], color: 0 for black, 1 for white

        if params['hand'] == 'black':
            # black
            if params["display_board"]:
                env.render()
            enables = np.array(env.possible_actions)
            action, probs = \
                (env.board_size ** 2 + 1, None) if len(enables) == 0 else agent.place(observation, enables, 0, t+1)
            observation, reward, done, _ = env.step([action, 0])
            loss = agent.learn((observation, action, reward, observation, done, enables))
            current_game_info.set_epoch(loss)
            if reward != 0: current_reward = reward

            # white
            if not done:
                if params["display_board"]:
                    env.render()
                enables = env.possible_actions
                action = [env.board_size ** 2 + 1 if len(enables) == 0 else random.choice(enables), 1]
                observation, reward, done, _ = env.step(action)
                if reward != 0: current_reward = reward

        elif params['hand'] == 'white':
            # black
            if params["display_board"]:
                env.render()
            enables = env.possible_actions
            action = [env.board_size ** 2 + 1 if len(enables) == 0 else random.choice(enables), 0]
            observation, reward, done, _ = env.step(action)
            if reward != 0: current_reward = reward

            # white
            if not done:
                if params["display_board"]:
                    env.render()
                enables = np.array(env.possible_actions)
                action, probs = \
                    (env.board_size ** 2 + 1, None) if len(enables) == 0 else agent.place(observation, enables, 0, t+1)
                observation, reward, done, _ = env.step([action, 1])
                loss = agent.learn((observation, action, reward, observation, done, enables))
                current_game_info.set_epoch(loss)
                if reward != 0: current_reward = reward

        if not done:
            # print("correct move")
            pass
        if done and params["display_each_winner"]:
            black_score = len(np.where(env.state[0, :, :] == 1)[0])
            white_score = len(np.where(env.state[1, :, :] == 1)[0])
            current_game_info.set_winner(black_score, white_score)
            current_game_info.set_reward(current_reward)
            current_game_info.set_turns(t)
            # if current_reward == -100:
            #     inval += 1
            #     current_game_info.set_enable_rankings(enables, probs)
            #     print(current_game_info.enable_ranking)
            #     print("invalid place, you lost; black-white-invalid: {}-{}-{}".format(
            #         black_win, white_win, inval))
            #     print('\n')
            if black_score > 32:
                black_win += 1
                print("black wins: {}:{}; black-white-invalid: {}-{}-{}".format(
                    black_score, white_score, black_win, white_win, inval))
            elif white_score > 32:
                white_win += 1
                print("white wins: {}:{}; black-white-invalid: {}-{}-{}".format(
                    black_score, white_score, black_win, white_win, inval))
            else:
                print(black_score, white_score)
                print("black-white-invalid: {}-{}-{}".format(black_win, white_win, inval))
            break
    agent.analyzer.insert_game_info(current_game_info)

    if i_episode != 0 and i_episode % 200 == 0:
        agent.agent_save_model(agent_name)

print("you won {} in {} games".format(white_win, params["max_epochs"]))
