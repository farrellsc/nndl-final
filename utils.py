class epochInfo:
    def __init__(self, loss):
        self.loss = loss


class gameInfo:
    def __init__(self):
        self.epochs = []
        self.reward = None
        self.turns = None
        self.winner = []      # black:white

        self.enable_ranking = {}

    def set_epoch(self, loss: float) -> None:
        current_epoch = epochInfo(loss)
        self.epochs.append(current_epoch)

    def set_reward(self, reward: int) -> None:
        self.reward = reward

    def set_winner(self, black_score: int, white_score: int) -> None:
        self.winner = [black_score, white_score]

    def set_turns(self, turns: int) -> None:
        self.turns = turns

    def set_enable_rankings(self, enables: list, probs: list) -> None:
        sorted_probs = [[i, probs[i]] for i in range(len(probs))]
        sorted_probs.sort(key=lambda x: x[1], reverse=True)
        sorted_probs = [[i]+sorted_probs[i] for i in range(len(probs))]
        for enable in enables:
            tup = list(filter(lambda x: x[1] == enable, sorted_probs))[0]
            self.enable_ranking[enable] = (tup[0], tup[2])
