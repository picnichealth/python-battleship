import numpy
import numpy as np

from battleship.board import Board, Square, empty_board
from battleship.bot import random_setup, random_move


def test_random_setup():
    np.random.seed(0)

    # normal setup on a normal board
    ship_sizes = [5, 4, 3, 3, 2]
    board = empty_board(10, 10)
    random_setup(ship_sizes, board)

    assert np.sum(board == Square.SHIP) == sum(ship_sizes), "all the ships were placed"

    assert np.all(
        (board == Square.EMPTY) | (board == Square.SHIP)
    ), "all squares are either empty or have ships"

    # a deterministic placement:
    # 3x1 board and ship of size 3
    board = empty_board(1, 3)
    random_setup([3], board)
    assert np.array_equal(board, [[Square.SHIP, Square.SHIP, Square.SHIP]])


def test_random_move():
    numpy.random.seed(0)

    # a 1x1 board
    board: Board = np.array([[Square.UNKNOWN]], dtype=int)  # type:ignore
    assert random_move(board) == (0, 0)

    # a board full of misses, except 1 square unknown
    board: Board = np.full((10, 10), Square.MISS, dtype=int)  # type:ignore
    board[3, 4] = Square.UNKNOWN
    assert (3, 4) == random_move(board)
