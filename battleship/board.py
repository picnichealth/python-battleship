from enum import IntEnum
from typing import Any, NamedTuple
import numpy as np


class Square(IntEnum):
  EMPTY = 0
  SHIP = 1
  MISS = 2
  HIT = 3
  UNKNOWN = 4

'''
Type representing a player's view of their own board.

These are 2-D numpy arrays with values in (Square.EMPTY, Square.SHIP, Square.MISS, Square.HIT).
Declared "Any" because numpy doesn't have type stubs yet.
'''
Board = Any

'''
Type representing a player's view of their opponent's board.

These are 2-D numpy arrays with values in (Square.UNKNOWN, Square.MISS, Square.HIT).
Declared "Any" because numpy doesn't have type stubs yet.
'''
OpponentBoard = Any


class Point(NamedTuple):
  ''' Row & column coordinates on a board. '''
  r: int
  c: int


def empty_board(rows: int, columns: int) -> Board:
  assert rows >= 1
  assert columns >= 1
  return np.full((rows, columns), Square.EMPTY, dtype=int)


def opponent_view(board: Board) -> OpponentBoard:
  '''
  Return a copy of this board as seen by the opponent.
  '''
  view = board.copy()
  view[view == Square.EMPTY] = Square.UNKNOWN
  view[view == Square.SHIP] = Square.UNKNOWN
  return view


def all_sunk(board: Board) -> bool:
  '''
  Are all the ships on the board sunk?
  '''
  return not np.any(board == Square.SHIP)


def num_shots(board: Board) -> int:
  '''
  Return how many shots have been fired.
  '''
  return board.size - np.sum(opponent_view(board) == Square.UNKNOWN)


def make_move(move: Point, board: Board) -> None:
   max_r, max_c = board.shape
   r, c = move

   if r < 0 or c < 0 or r >= max_r or c >= max_c:
     raise ValueError(f'{move} out of range.')

   if board[r, c] == Square.SHIP:
     board[r, c] = Square.HIT
   elif board[r, c] == Square.EMPTY:
     board[r, c] = Square.MISS
   else:
     raise ValueError(f'{move} already moved on board {board}')
