from typing import List

from battleship.board import Board, OpponentBoard, Point, Square
from battleship.bot import random_move, random_setup
from battleship.interface import print_boards
from battleship.player import Player


def is_hit(opponent_board: OpponentBoard, move: Point) -> bool:
    return opponent_board[move.r, move.c] == Square.HIT


def point_range(start_point: Point, delta_r: int, delta_c: int, board: Board):
    """
    Starting at start_point, and moving (delta_r, delta_c) squares at each step,this generator
    yields one point for every square until the edge of the board is detected or max_steps is
    reached.

    We do NOT yield start_point itself.
    """
    r, c = start_point.r, start_point.c
    assert delta_r != 0 or delta_c != 0

    while True:
        r += delta_r
        c += delta_c
        if not (0 <= r < board.shape[0] and 0 <= c < board.shape[1]):
            break
        yield Point(r, c)


class RulesBot(Player):
    def __init__(self):
        self.strategy_coro = self.get_strategy_coroutine()

        # Coroutine needs to have send(None) to execute the routine up-until the
        # first yield.
        self.strategy_coro.send(None)

    def setup(self, ship_sizes: List[int], board: Board) -> None:
        return random_setup(ship_sizes, board)

    def choose_move(self, opponent_board: OpponentBoard) -> Point:
        return self.strategy_coro.send(opponent_board)

    def display(self, board: Board, opponent_board: OpponentBoard) -> None:
        # don't display anything from the bot's perspective
        print_boards(board, opponent_board)

    def get_strategy_coroutine(self):
        # Strategy coroutine.
        opponent_board = yield
        move = None

        while True:
            # Phase 1:  Search mode
            last_move_hit = False
            while not last_move_hit:
                move = random_move(opponent_board)
                opponent_board = yield move
                last_move_hit = is_hit(opponent_board, move)

            # Phase 2: Destroy
            for direction in (-1, 0), (1, 0), (0, -1), (0, 1):
                for point in point_range(move, *direction, opponent_board):
                    state = opponent_board[point.r, point.c]
                    if state == Square.UNKNOWN:
                        # We must explore this square.
                        opponent_board = yield point
                        state = opponent_board[point.r, point.c]

                    if state == Square.MISS:
                        break  # We found the end for this direction
