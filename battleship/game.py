import os
import sys

from abc import ABC, abstractmethod
from typing import Callable, List, Tuple, NamedTuple

import numpy as np
import tensorflow as tf

from battleship.board import Point, Board, OpponentBoard, Square, empty_board, opponent_view, all_sunk, make_move, num_shots
from battleship.bot import apply_random_setup, random_move
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
  def choose_move(self, opponent_board: OpponentBoard) -> Point:
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
    return apply_random_setup(ship_sizes, board)

  def choose_move(self, opponent_board: OpponentBoard) -> Point:
    return input_move(opponent_board)

  def display(self, board: Board, opponent_board: OpponentBoard) -> None:
    print_boards(board, opponent_board)

class RandomBot(Player):
  '''
  This starting impelementation uses a random setup, and moves randomly.
  '''
  def setup(self, ship_sizes: List[int], board: Board) -> None:
    return apply_random_setup(ship_sizes, board)

  def choose_move(self, opponent_board: OpponentBoard) -> Point:
    return random_move(opponent_board)

  def display(self, board: Board, opponent_board: OpponentBoard) -> None:
    # don't display anything from the bot's perspective
    pass

class ProbBot(Player):
  '''
  This implementation uses a pre-trained model to estimate ship location
  probability density based on observed hits and misses.
  '''

  def __init__(self, model):
    self.model = model

  def setup(self, ship_sizes: List[int], board: Board) -> None:
    return apply_random_setup(ship_sizes, board)

  def choose_move(self, opponent_board: OpponentBoard) -> Point:

    # predict the probability density for ships based on the current
    # view of the opponents board
    probs = self.model.predict(opponent_board.reshape(1, ROWS, COLUMNS, 1))

    probs = np.squeeze(probs)

    # filter out coordinates we've already fired
    next_best_val = probs[opponent_board == Square.UNKNOWN].max()

    # get the coordinates of the most likely spot(s)
    next_best_coords = np.concatenate(np.where(probs == next_best_val)).reshape(-1, 2)

    return Point(*next_best_coords[0])

  def display(self, board: Board, opponent_board: OpponentBoard) -> None:
    print_boards(board, opponent_board)


def play_one_game(player1: Player, player2: Player, display=True):
  player1_board = empty_board(ROWS, COLUMNS)
  player2_board = empty_board(ROWS, COLUMNS)

  player1.setup(SHIP_SIZES, player1_board)
  player2.setup(SHIP_SIZES, player2_board)

  while True:
    if display:
      player1.display(player1_board, opponent_view(player2_board))
    move = player1.choose_move(opponent_view(player2_board))
    make_move(move, player2_board)

    if all_sunk(player2_board):
      print("🎉 Player 1 wins! 🎉")
      return (1, num_shots(player2_board))

    if display:
      player2.display(player2_board, opponent_view(player1_board))
    move = player2.choose_move(opponent_view(player1_board))
    make_move(move, player1_board)

    if all_sunk(player1_board):
      print("🏴‍☠️ Player 2 wins! 🏴‍☠️")
      return (2, num_shots(player1_board))


if __name__ == '__main__':
  '''
  This starting implementation hardcodes player 1 as human and player 2 as a bot.
  '''

  this_dir = (os.path.dirname(os.path.realpath(__file__)))
  model = tf.keras.models.load_model(os.path.join(this_dir, 'models/demo_model'))

  play_one_game(ProbBot(model), RandomBot())
