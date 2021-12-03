import os
import numpy as np

from battleship.board import Point, Square, Board, OpponentBoard

COLUMN_NAMES = "abcdefghijklmnopqrstuvwxyz"

# fullwidth unicode
#
# assumes these are twice as wide as ascii in the display.
# if your terminal breaks that assumption or lacks unicode support,
# you can replace each with 2 ascii characters.
SPRITES = {
    Square.EMPTY: "．",
    Square.UNKNOWN: "．",
    Square.SHIP: "🚢",
    Square.HIT: "💥",
    Square.MISS: "💦",
}


def print_boards(board: Board, opponent_board: OpponentBoard):
    # clear the terminal, if running on a unix-like system
    try:
        os.system("clear")
    except:
        pass

    rows, columns = board.shape
    if columns > len(COLUMN_NAMES):
        raise ValueError(
            "This interface uses letters for columns, so it can't handle boards that large."
        )

    board_display_width = 3 * columns
    horizontal_bar = (
        "---+" + "-" * board_display_width + "+" + "-" * board_display_width + "+"
    )

    print("")
    print(
        "   | YOUR BOARD"
        + " " * (board_display_width - len(" YOUR BOARD"))
        + "| ENEMY BOARD"
    )
    print(horizontal_bar)
    # column labels
    print("   |", end="")
    for c in range(columns):
        print(f" {COLUMN_NAMES[c]} ", end="")
    print("|", end="")
    for c in range(columns):
        print(f" {COLUMN_NAMES[c]} ", end="")
    print("|")
    print(horizontal_bar)

    for r in range(rows):
        # row label
        print(f" {r+1: >2}|", end="")

        # our board
        for c in range(columns):
            print(SPRITES[board[r, c]], end=" ")

        print("|", end="")

        for c in range(columns):
            print(SPRITES[opponent_board[r, c]], end=" ")

        print("|")

    print(horizontal_bar)
    print("")


def input_move(opponent_board: OpponentBoard) -> Point:
    max_r, max_c = opponent_board.shape

    while True:
        raw_input = input('Move (e.g "f1") --> ')
        try:
            col = raw_input[0]
            row = raw_input[1:]
            r = int(row) - 1
            c = COLUMN_NAMES.index(col.lower())

            if r < 0 or c < 0 or r >= max_r or c >= max_c:
                print("Out of bounds.")
            elif opponent_board[r, c] != Square.UNKNOWN:
                print("Already shot there.")
            else:
                return Point(r, c)
        except:
            print("Input not recognized.")
            pass
