import torch
from torch import nn

from battleship.board import Square


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
last_conv_channels = 30
model = nn.Sequential(
    Unsqueeze(),
    nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding='same'),
    nn.ReLU(),
    nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3, 3), padding='same'),
    nn.ReLU(),
    nn.Conv2d(in_channels=20, out_channels=last_conv_channels, kernel_size=(3, 3), padding='same'),
    nn.ReLU(),
    nn.Flatten(),

    nn.Linear(last_conv_channels * 10 * 10, 10 * 10),
    FixShape(),
)


def get_model_input(board):
    # Create a representation of what we know about the opponents board.
    # This is similar to the OpponentBoard representation used by the game engine, but we map the
    # values to match what the model expects:
    # Square.HIT gets mapped to 0
    # Square.MISS gets mapped to 1
    # Square.UNKNOWN gets mapped to 2

    for r in range(10):
        for c in range(10):
            if board[r, c] == Square.HIT:
                board[r, c] = 0
            elif board[r, c] == Square.MISS:
                board[r, c] = 1
            elif board[r, c] == Square.UNKNOWN:
                board[r, c] = 2

    return board

