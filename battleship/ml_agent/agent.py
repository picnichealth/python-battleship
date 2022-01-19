import torch
from battleship.ml_agent.model import model, get_model_input
from typing import List

from battleship.board import Board, OpponentBoard, Point, Square
from battleship.bot import random_setup
from battleship.interface import print_boards
from battleship.player import Player


# Load model
state_dict = torch.load("model.pt")
model.load_state_dict(state_dict)


class MLBot(Player):
    def setup(self, ship_sizes: List[int], board: Board) -> None:
        return random_setup(ship_sizes, board)

    def choose_move(self, opponent_board: OpponentBoard) -> Point:
        opponent_board = torch.tensor(opponent_board)

        model_input = get_model_input(opponent_board)
        model_input = model_input.unsqueeze(0) # Add batch dim of 1

        pred_occupancy = model(model_input)[0]
        # Pred occupancy now contains the models prediction of which squares
        # are occupied

        # Let's mask out all squares that are not unknown
        pred_occupancy = torch.where(
            opponent_board == Square.UNKNOWN,
            torch.sigmoid(pred_occupancy),
            torch.zeros_like(pred_occupancy)
        )

        # Now lets find the best square
        r, c = (pred_occupancy == pred_occupancy.max()).nonzero()[0]

        return Point(r, c)


    def display(self, board: Board, opponent_board: OpponentBoard) -> None:
        # don't display anything from the bot's perspective
        print_boards(board, opponent_board)
