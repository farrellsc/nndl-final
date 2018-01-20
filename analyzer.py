from utils import *
import matplotlib.pyplot as plt
import numpy as np


class analyzer:
    def __init__(self):
        self.game_records = []
        self.analyzer_dir = "./analyze/"

    def insert_game_info(self, game_info: gameInfo) -> None:
        self.game_records.append(game_info)

    def plot_winning_rate(self, file_name_no_suffix: str) -> None:
        plt.bar(range(len(self.game_records)), [game.winner[1]/sum(game.winner)-0.5 for game in self.game_records])
        plt.ylim([-0.5, 0.5])
        plt.title("winning rate")
        plt.savefig(self.analyzer_dir + file_name_no_suffix + '/winning_rate.jpg')
        plt.close()

    def plot_winning_ratio(self, file_name_no_suffix: str) -> None:
        wins = [1 if game.winner[1] > 32 else 0 for game in self.game_records]
        wins = wins[:100*int(len(wins)/100)]
        wins = np.sum(np.reshape(wins, [-1, 100]), axis=1)
        plt.bar(range(len(wins)), wins)
        plt.ylim([0, 100])
        plt.title('winning ratio by bins')
        plt.savefig(self.analyzer_dir + file_name_no_suffix + '/winning_ratio.jpg')
        plt.close()

    def plot_loss(self, file_name_no_suffix: str) -> None:
        losses = [epoch_info.loss for game in self.game_records for epoch_info in game.epochs]
        plt.bar(range(len(losses)), losses)
        plt.title("loss")
        plt.savefig(self.analyzer_dir + file_name_no_suffix + '/loss.jpg')
        plt.close()

    def plot_reward(self, file_name_no_suffix: str) -> None:
        rewards = [game.reward for game in self.game_records]
        plt.bar(range(len(rewards)), rewards)
        plt.title("reward")
        plt.ylim([-2, 1])
        plt.savefig(self.analyzer_dir + file_name_no_suffix + '/reward.jpg')
        plt.close()

    def plot_turns(self, file_name_no_suffix: str) -> None:
        turns = [game.turns for game in self.game_records]
        plt.bar(range(len(turns)), turns)
        plt.title("turns")
        plt.savefig(self.analyzer_dir + file_name_no_suffix + '/turns.jpg')
        plt.close()

