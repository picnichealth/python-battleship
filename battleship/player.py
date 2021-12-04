from abc import ABC, abstractmethod
from typing import List

from battleship.board import Point, Board, OpponentBoard


class Player(ABC):
    @abstractmethod
    def setup(self, ship_sizes: List[int], board: Board) -> None:
        """Modifies the board by placing the ship."""
        raise NotImplementedError()

    @abstractmethod
    def choose_move(self, opponent_board: OpponentBoard) -> Point:
        """Returns the coordinates of the next shot."""
        raise NotImplementedError()

    @abstractmethod
    def display(self, board: Board, opponent_board: OpponentBoard) -> None:
        """Display the board state from this player's perspective."""
        raise NotImplementedError()
