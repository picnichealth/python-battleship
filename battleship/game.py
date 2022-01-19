from battleship.board import empty_board, opponent_view, all_sunk, make_move
from battleship.bot import RandomBot
from battleship.interface import Human
from battleship.ml_agent.agent import MLBot
from battleship.rules_bot import RulesBot
from battleship.player import Player

ROWS = 10
COLUMNS = 10

SHIP_SIZES = [
    5,  # carrier
    4,  # battleship
    3,  # cruiser
    3,  # submarine
    2,  # destroyer
]


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


if __name__ == "__main__":
    """
    This starting implementation hardcodes player 1 as human and player 2 as a bot.
    """
    play_one_game(RulesBot(), MLBot())
