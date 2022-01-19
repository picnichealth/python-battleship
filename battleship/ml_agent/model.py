import torch
from torch import nn

from battleship.board import OpponentBoard, Square


class Unsqueeze(nn.Module):
    def forward(self, input):
        return torch.unsqueeze(input,1)

class Squeeze(nn.Module):
    def forward(self, input):
        return torch.squeeze(input, 1)

model = nn.Sequential(
    Unsqueeze(),
    nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding='same'),
    nn.ReLU(),
    nn.Conv2d(in_channels=10, out_channels=40, kernel_size=(3, 3), padding='same'),
    nn.ReLU(),
    nn.Conv2d(in_channels=40, out_channels=80, kernel_size=(3, 3), padding='same'),
    nn.ReLU(),
    nn.Conv2d(in_channels=80, out_channels=160, kernel_size=(3, 3), padding='same'),
    nn.ReLU(),
    nn.Conv2d(in_channels=160, out_channels=1, kernel_size=(1, 1), padding='same'),
   # nn.Sigmoid(),
    Squeeze(),
)


def get_model_input(opponent_board: OpponentBoard):
    # Create a representation what we know about the board state.
    # We'll use a 10x10 array, where
    # 0 represents a miss
    # 1 represents a hit
    # -1 represents unknown

    board = torch.zeros((10,10), dtype=torch.float32)
    for r in range(10):
        for c in range(10):
            if opponent_board[r, c] == Square.HIT:
                board[r,c] = 1
            elif opponent_board[r, c] == Square.MISS:
                board[r,c] = 0
            elif opponent_board[r, c] == Square.UNKNOWN:
                board[r,c] = -1

    return board

