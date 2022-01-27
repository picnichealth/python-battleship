from battleship.board import empty_board, opponent_view, all_sunk, make_move
from battleship.bot import RandomBot, random_setup
from battleship.rules_bot import RulesBot
from battleship.player import Player
import numpy as np

GAME_COUNT = 200
ROWS = 10
COLUMNS = 10

SHIP_SIZES = [
    5,  # carrier
    4,  # battleship
    3,  # cruiser
    3,  # submarine
    2,  # destroyer
]


def play_one_game(player: Player):
    board = empty_board(ROWS, COLUMNS)
    random_setup(SHIP_SIZES, board)

    move_cnt = 0
    while not all_sunk(board):
        move = player.choose_move(opponent_view(board))
        make_move(move, board)
        move_cnt += 1
    return move_cnt


def eval_bot(bot: Player):
    results = [play_one_game(bot()) for i in range(GAME_COUNT)]
    return np.mean(results)


if __name__ == "__main__":
    for bot_name, bot in [('Random', RandomBot), ('RulesBot', RulesBot)]:
        print(f"Average number of turns to sink random board using {bot_name}: {eval_bot(bot)}")

