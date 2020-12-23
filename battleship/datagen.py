import numpy as np

from battleship.board import empty_board, opponent_view, Square
from battleship.bot import generate_random_setup


def generate_random_boards(n, rows, columns, ship_sizes):
    """Generate N random board layouts as an (n, rows, columns) array"""

    empty = empty_board(rows, columns)

    boards = [generate_random_setup(ship_sizes, empty) for _ in range(n)]

    return np.stack(boards)


def generate_shots(n, board):
    """Generate n randomly distributed shots within board"""

    shots = np.concatenate([[1] * n, [0] * (board.size - n)])
    np.random.shuffle(shots)

    return shots.reshape(*board.shape)


def generate_shot_results(board, shots):
    """Generate an opponent's view of applying shots to a board"""

    # make a copy so that we don't modify the board in place and
    # can use it again later
    board = board.copy()

    # hits are places where both board = 1 and shots = 1
    # so element-wise multiplication finds the intersection
    hits = np.multiply(board, shots)

    misses = shots - hits

    board[hits == 1] = Square.HIT
    board[misses == 1] = Square.MISS

    return opponent_view(board)


def probs_given_shot_results(sample_boards, shot_results):
    """Estimate probability of ship presence given some shot results

    Parameters
    ----------
    sample_boards: np.ndarray
        An array of shape (N, ROWS, COLUMNS) containing N randomly generated
        board layouts.
    shot_results: np.ndarray
        An array of shape (ROWS, COLUMNS) with values in (Square.HIT,
        Square.MISS, and Square.UNKNOWN) containing observed hits and misses
    """

    # generate (row, column)-shaped boolean masks for the observed hits and
    # misses
    hit_mask = shot_results == Square.HIT
    miss_mask = shot_results == Square.MISS

    # check that for each sample board, each square where we observe a hit has
    # a ship and each square where we observe a miss is empty
    hit_possible = (sample_boards[:, hit_mask] == Square.SHIP).all(axis=1)
    miss_possible = (sample_boards[:, miss_mask] == Square.EMPTY).all(axis=1)

    # filter the samples to only those which are compatible with the observed
    # hits and misses
    possible = sample_boards[hit_possible & miss_possible, :, :]

    # sum over all sample boards which are possible given the observed shot
    # results
    flattened = possible.sum(axis=0)

    # normalize the output so training examples all have the same scale
    if flattened.max() > 0:
        return flattened / flattened.max()
    else:
        return flattened


def generate_training_data(
    n_train_samples, sample_boards, rows, columns, ship_sizes, max_shots=15
):
    """Generate a set of training data"""

    def _generator():
        # generate a single random board and a set of shots
        board = generate_random_setup(ship_sizes, empty_board(rows, columns))
        shots = generate_shots(np.random.randint(1, max_shots), board)

        X = generate_shot_results(board, shots)
        y = probs_given_shot_results(sample_boards, X)

        return X, y

    # take n_train_samples from the generator then convert list of tuples to
    # tuple of lists
    samples = [_generator() for _ in range(n_train_samples)]
    Xs, ys = zip(*samples)

    # add an extra dimension on Xs and ys so that they have a single "channel"
    # for subsequent modeling. Both Xs and ys have shape
    # (n_train_samples, rows, columns, 1)
    Xs = np.expand_dims(np.array(Xs), -1)
    ys = np.expand_dims(np.array(ys), -1)

    return Xs, ys
