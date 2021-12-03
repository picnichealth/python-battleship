# Implementation of a simple rules-based battleship bot.
# The bot will start in search mode.  In this mode, it randomly shoots at the board until it hits
# a ship.  After the first hit it should changes to destroy mode.
#
# In destroy mode, it loops through four directions: up, down, left, and right.
# For each direction it will continue shooting until it gets a miss or runs out of room on the board.
#
# After completing the destroy process in all four directions it returns to search mode.
#
#
import random
from typing import List

from battleship.board import Board, OpponentBoard, Point, Square
from battleship.bot import random_setup
from battleship.interface import print_boards
from battleship.player import Player
import numpy as np


def get_good_move(opponent_board: OpponentBoard):
    rows, cols = opponent_board.shape

    miss_count = np.zeros_like(opponent_board)

    for r in range(rows):
        for c in range(cols):
            if opponent_board[r, c] != Square.UNKNOWN:
                miss_count[r, c] = 100  # Remove this option from consideration
                continue

            miss = 0
            if (r-1) >= 0 and opponent_board[r-1, c] == Square.MISS:
                miss += 1

            if (r+1) < rows and opponent_board[r+1, c] == Square.MISS:
                miss += 1

            if (c-1) >= 0 and opponent_board[r, c-1] == Square.MISS:
                miss += 1

            if (c+1) < cols and opponent_board[r, c+1] == Square.MISS:
                miss += 1

            miss_count[r, c] = miss
    best_rows, best_cols = np.where(miss_count == miss_count.min())
    r, c = random.choice(list(zip(best_rows, best_cols)))
    return Point(r, c)


def get_square_safe(opponent_board: OpponentBoard, r: int, c: int) -> Square:
    rows, cols = opponent_board.shape

    if 0 <= r < rows and 0 <= c < cols:
        return opponent_board[r,c]
    else:
        return Square.MISS


class RulesBot(Player):
    def setup(self, ship_sizes: List[int], board: Board) -> None:
        return random_setup(ship_sizes, board)

    def choose_move(self, opponent_board: OpponentBoard) -> Point:
        rows, cols = opponent_board.shape

        # Loop through the board looking for any destroy mode moves!
        for r in range(rows):
            for c in range(cols):
                if opponent_board[r, c] != Square.HIT:
                    continue

                # We're now looping through all HITS
                # Let's get the state of our neighbors for convenience.
                square_l = get_square_safe(opponent_board, r - 1, c)
                square_r = get_square_safe(opponent_board, r + 1, c)
                square_u = get_square_safe(opponent_board, r, c - 1)
                square_d = get_square_safe(opponent_board, r, c + 1)

                if not (square_l == Square.HIT or
                        square_r == Square.HIT or
                        square_u == Square.HIT or
                        square_d == Square.HIT):
                    # We have a single hit with no neighboring hits
                    # This means we're at the beginning of the destroy phase, and we still need
                    # to figure out the direction of the ship.

                    if square_l == Square.UNKNOWN:
                        return Point(r - 1, c)
                    elif square_r == Square.UNKNOWN:
                        return Point(r + 1, c)
                    elif square_u == Square.UNKNOWN:
                        return Point(r, c - 1)
                    elif square_d == Square.UNKNOWN:
                        return Point(r, c + 1)

                # Assuming we don't have any size-1 ships, getting here means that we have
                # two hits in a row somewhere on board, and we need to continue the pattern of
                # hits.

                if square_l == Square.HIT and square_r == Square.UNKNOWN:
                    return Point(r + 1, c)

                if square_r == Square.HIT and square_l == Square.UNKNOWN:
                    return Point(r - 1, c)

                if square_u == Square.HIT and square_d == Square.UNKNOWN:
                    return Point(r, c + 1)

                if square_d == Square.HIT and square_u == Square.UNKNOWN:
                    return Point(r, c - 1)

        # If we get here we have no destroy mode moves -- switch to search mode
        return get_good_move(opponent_board)

    def display(self, board: Board, opponent_board: OpponentBoard) -> None:
        # don't display anything from the bot's perspective
        print_boards(board, opponent_board)
