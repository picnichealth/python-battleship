import random
from typing import List, NamedTuple

import numpy as np

from battleship.board import Board, OpponentBoard, Point, Square
from battleship.player import Player


class _Placement(NamedTuple):
    """A top-left corner and orientation of a ship within a board."""

    r: int
    c: int
    is_vertical: bool


def _all_valid_placements(ship_size: int, board: Board) -> List[_Placement]:
    max_r, max_c = board.shape
    placements: List[_Placement] = []

    # try each point as the top-left corner
    for r in range(max_r):
        for c in range(max_c):
            # try vertical
            if r + ship_size <= max_r and np.all(
                board[r : r + ship_size, c] == Square.EMPTY
            ):
                placements.append(_Placement(r, c, is_vertical=True))

            # try horizontal
            if c + ship_size <= max_c and np.all(
                board[r, c : c + ship_size] == Square.EMPTY
            ):
                placements.append(_Placement(r, c, is_vertical=False))

    return placements


def random_setup(ship_sizes: List[int], board: Board) -> None:
    """
    Randomly place all ships onto the board, or throw an error if they don't fit.

    This method first enumerates all valid placements, and then chooses one randomly.

    Rejection sampling would be simpler, but some board sizes will have no valid placement,
    and we should loop fail with an error instead of looping infinitely.
    """
    for size in ship_sizes:

        valid_placements = _all_valid_placements(size, board)

        if not valid_placements:
            raise AssertionError(
                f"Ship with size {size} doesn't fit on the board: {board}"
            )

        r, c, is_vertical = random.choice(valid_placements)

        if is_vertical:
            board[r : r + size, c] = Square.SHIP
        else:
            board[r, c : c + size] = Square.SHIP


def random_move(opponent_board: OpponentBoard) -> Point:
    """
    Randomly choose an empty square to shoot at.
    """
    empty_rows, empty_columns = np.where(opponent_board == Square.UNKNOWN)

    choice = random.randint(0, len(empty_rows) - 1)

    return Point(r=empty_rows[choice], c=empty_columns[choice])


class RandomBot(Player):
    """
    This starting impelementation uses a random setup, and moves randomly.
    """

    def setup(self, ship_sizes: List[int], board: Board) -> None:
        return random_setup(ship_sizes, board)

    def choose_move(self, opponent_board: OpponentBoard) -> Point:
        return random_move(opponent_board)

    def display(self, board: Board, opponent_board: OpponentBoard) -> None:
        # don't display anything from the bot's perspective
        pass
