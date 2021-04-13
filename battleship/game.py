import sys
from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, NamedTuple
import numpy as np
import random

from battleship.board import (Point, Board, OpponentBoard, Square,
                              empty_board, opponent_view, all_sunk, make_move)
from battleship.bot import random_setup, random_move
from battleship.interface import input_move, print_boards

ROWS = 10
COLUMNS = 10

SHIP_SIZES = [
  5, # carrier
  4, # battleship
  3, # cruiser
  3, # submarine
  2, # destroyer
]

class Player(ABC):

  @abstractmethod
  def setup(self, ship_sizes: List[int], board: Board) -> None:
    ''' Modifies the board by placing the ship. '''
    raise NotImplementedError()

  @abstractmethod
  def choose_move(self, opponent_board: OpponentBoard):
    ''' Returns the coordinates of the next shot. '''
    raise NotImplementedError()

  @abstractmethod
  def display(self, board: Board, opponent_board: OpponentBoard) -> None:
    ''' Display the board state from this player's perspective. '''
    raise NotImplementedError()


class Human(Player):
  '''
  This starting implementation uses a random setup, and moves based
  on user input.
  '''
  def setup(self, ship_sizes: List[int], board: Board) -> None:
    return random_setup(ship_sizes, board)

  def choose_move(self, opponent_board: OpponentBoard) -> Point:
    return input_move(opponent_board)

  def display(self, board: Board, opponent_board: OpponentBoard) -> None:
    print_boards(board, opponent_board)


class RandomBot(Player):
  '''
  This starting implementation uses a random setup, and moves randomly.
  '''
  def setup(self, ship_sizes: List[int], board: Board) -> None:
    return random_setup(ship_sizes, board)

  def choose_move(self, opponent_board: OpponentBoard) -> Point:
    return random_move(opponent_board)

  def display(self, board: Board, opponent_board: OpponentBoard) -> None:
    # don't display anything from the bot's perspective
    pass

def play_one_game(player1: Player, player2: Player):
  player1_board = empty_board(ROWS, COLUMNS)
  player2_board = empty_board(ROWS, COLUMNS)

  player1.setup(SHIP_SIZES, player1_board)
  player2.setup(SHIP_SIZES, player2_board)

  while True:
    player1.display(player1_board, opponent_view(player2_board))
    move = player1.choose_move(opponent_view(player2_board))
    make_move(move, player2_board)

    if all_sunk(player2_board):
      print("🎉 Player 1 wins! 🎉")
      return

    player2.display(player2_board, opponent_view(player1_board))
    move = player2.choose_move(opponent_view(player1_board))
    make_move(move, player1_board)

    if all_sunk(player1_board):
      print("🏴‍☠️ Player 2 wins! 🏴‍☠️")
      return


if __name__ == '__main__':
  '''
  This starting implementation hardcodes player 1 as human and player 2 as a bot.
  '''
  play_one_game(Human(), RandomBot())
