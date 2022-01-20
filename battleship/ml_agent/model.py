import torch
from torch import nn

from battleship.board import OpponentBoard, Square


# Define a few layers for convenience
class Unsqueeze(nn.Module):
    def forward(self, input):
        return torch.unsqueeze(input,1)

class PrintShape(nn.Module):
    def forward(self, input):
        print(input.shape)
        return input

class FixShape(nn.Module):
    def forward(self, input):
        n_batch = input.shape[0]
        return torch.reshape(input, (n_batch, 10, 10))


# Our model will be a CNN.  This will allow the network to learn a translational invariant solution.
# The model will take as input a representation of our knowledge of the opponent board, and will
# return as output the probability of any given square being occupied.
model = nn.Sequential(
    Unsqueeze(),
    nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding='same', bias=False),
    nn.ReLU(),
    nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding='same', bias=False),
    nn.ReLU(),
    nn.Conv2d(in_channels=20, out_channels=30, kernel_size=(3, 3), padding='same', bias=False),
    nn.ReLU(),
    nn.Flatten(),

    # These linear layers seem to really help the model reduce the training loss
    nn.Linear(3000, 1024),
    nn.ReLU(),
    nn.Linear(1024, 100),
    FixShape(),
)


def get_model_input(opponent_board: OpponentBoard):
    # Create a representation of what we know about the opponents board.
    # We'll use a 10x10 array, where
    # 0 represents a miss
    # 1 represents a hit
    # -1 represents unknown

    board = torch.zeros((10, 10), dtype=torch.float32)
    for r in range(10):
        for c in range(10):
            if opponent_board[r, c] == Square.HIT:
                board[r, c] = 1
            elif opponent_board[r, c] == Square.MISS:
                board[r, c] = 0
            elif opponent_board[r, c] == Square.UNKNOWN:
                board[r, c] = -1

    return board

