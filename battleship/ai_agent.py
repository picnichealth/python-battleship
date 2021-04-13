import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


from typing import Callable, List, Tuple, NamedTuple
from battleship.game import ROWS, COLUMNS, Player, SHIP_SIZES, RandomBot
from battleship.board import (Point, Board, OpponentBoard, Square,
                              empty_board, opponent_view, all_sunk, make_move)
from battleship.bot import random_setup, random_move
from battleship.interface import input_move, print_boards


hidden_size = ROWS*COLUMNS*5
learning_rate = 2e-4
weight_decay = 1e-5

reduce_rate = 0.95

log_every = 10


class SlidingAverage:
    def __init__(self, name, steps=10):
        self.name = name
        self.steps = steps
        self.t = 0
        self.ns = []
        self.avgs = []

    def add(self, n):
        self.ns.append(n)
        if len(self.ns) > self.steps:
            self.ns.pop(0)
        self.t += 1
        if self.t % self.steps == 0:
            self.avgs.append(self.value)

    @property
    def value(self):
        if len(self.ns) == 0: return 0
        return sum(self.ns) / len(self.ns)

    def __str__(self):
        return "%s=%.4f" % (self.name, self.value)

    def __gt__(self, value):
        return self.value > value

    def __lt__(self, value):
        return self.value < value


def masked_softmax(vec, mask, dim=1, epsilon=1e-5):
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return (masked_exps/masked_sums)


class Policy(nn.Module):
    def __init__(self, hidden_size):
        super(Policy, self).__init__()
        # the agent is able to see all
        visible_squares = ROWS*COLUMNS
        input_size = visible_squares

        self.inp1 = nn.Linear(input_size, hidden_size)
        self.inp2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, ROWS*COLUMNS , bias=False)
        # For actions

    def forward(self, x):
        x = x.view(1, -1)
        x = torch.tanh(x)  # Squash inputs
        x = F.relu(self.inp1(x))
        x = F.relu(self.inp2(x))
        x = self.out(x)
        return x


class Policy2d(nn.Module):
    def __init__(self):
        super(Policy2d, self).__init__()
        # the agent is able to see all
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.out = nn.Linear(ROWS*COLUMNS*32, ROWS*COLUMNS , bias=False)
        # For actions

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


class Environment(Player):
    """
    This starting implementation of an AI agent.
    """
    def __init__(self):
        self.policy = Policy(hidden_size=hidden_size)
        #self.policy = Policy2d()
        self.pos_memory = []
        self.action_memory = []
        self.m_memory = []

    def setup(self, ship_sizes: List[int], board: Board) -> None:
        # clear history when start each game
        self.pos_memory = []
        self.action_memory = []
        self.m_memory = []
        return random_setup(ship_sizes, board)

    def choose_move(self, opponent_board: OpponentBoard):
        state = Variable(torch.from_numpy(opponent_board.flatten()).float())
        #state = Variable(torch.from_numpy(
        #    np.expand_dims(opponent_board, axis=(0, 1))).float())
        scores = self.policy(state)  # Forward state through network
        # maskout the impossible place

        mask = torch.from_numpy((opponent_board == Square.UNKNOWN).
                                flatten()).float()
        probs = masked_softmax(scores, mask)
        # sample action based on probability
        m = torch.distributions.Categorical(probs)
        action = m.sample()

        r, c = np.unravel_index(action.item(), opponent_board.shape)
        point = Point(r, c)

        self.action_memory.append(action)
        self.m_memory.append(m)
        self.pos_memory.append(point)
        return point, m, action

    def display(self, board: Board, opponent_board: OpponentBoard) -> None:
        print_boards(board, opponent_board)


def get_reward(opponent_board: OpponentBoard, pos: Point):
    """
    generate score of each shot
    :param opponent_board:
    :param pos:
    :return:
    """
    r,c = pos
    if opponent_board[r][c] == Square.HIT:
        reward = 5
    else:
        reward = -1
    return reward


# ## Playing through an episode

def play_one_game(player1: Player, player2: Player):
    rewards = []
    player1_board = empty_board(ROWS, COLUMNS)
    player2_board = empty_board(ROWS, COLUMNS)

    player1.setup(SHIP_SIZES, player1_board)
    player2.setup(SHIP_SIZES, player2_board)
    while True:
        # conduct reinforcement learning
        #player1.display(player1_board, opponent_view(player2_board))
        if len(player1.pos_memory) == 0:
            move = random_move(opponent_view(player2_board))
            make_move(move, player2_board)
            reward = get_reward(opponent_view(player2_board), move)
            player1.pos_memory.append(move)
        else:
            move, m,  action = player1.choose_move(
                opponent_view(player2_board))
            make_move(move, player2_board)
            reward = get_reward(opponent_view(player2_board), move)

            # policy gradient
            loss = -m.log_prob(action) * reward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        rewards.append(reward)
        if all_sunk(player2_board):
            print("🎉 Player 1 wins! 🎉")
            return rewards

        #player2.display(player2_board, opponent_view(player1_board))
        move = player2.choose_move(opponent_view(player1_board))
        make_move(move, player1_board)

        if all_sunk(player1_board):
            print("🏴‍☠️ Player 2 wins! 🏴‍☠️")
            return rewards


if __name__ == '__main__':
    """
    """
    e = 0
    ai_agent = Environment()
    random_boat = RandomBot()
    optimizer = optim.SGD(ai_agent.policy.parameters(), lr=learning_rate,
                          momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5,
                                                gamma=reduce_rate)
    reward_avg = SlidingAverage('reward avg', steps=log_every)
    final_reward_records = []
    print("Training start")
    while e < 10000:
        rewards = play_one_game(ai_agent, random_boat)
        scheduler.step()
        all_reward = sum(rewards)

        final_reward_records.append((e, all_reward))
        reward_avg.add(all_reward)

        if e % log_every == 0:
            print('[epoch=%d]' % e, reward_avg)

        e += 1

    print("Testing start")
    play_one_game(ai_agent, random_boat)
