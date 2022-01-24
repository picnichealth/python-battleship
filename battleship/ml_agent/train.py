import sys

from torch import optim, nn
from torch.nn import functional as F
from battleship.board import empty_board, Square
from battleship.bot import random_setup
from tqdm import tqdm
import torch
import random
from battleship.ml_agent.model import model


# These are redefined here (also defined in game.py)
# The reason is that game.py imports agent.py, and agent.py loads the weights that we generate
# in this script.  So to avoid the circular dependency, we redefine the values here.
ROWS = 10
COLUMNS = 10

SHIP_SIZES = [
    5,  # carrier
    4,  # battleship
    3,  # cruiser
    3,  # submarine
    2,  # destroyer
]

def make_examples(board_cnt: int):
    # Generate random boards.

    # For each board we'll create a "board input", which represent our knowledge of an opponent
    # board and a "board output", which represents which cells are occupied.
    boards_input = torch.zeros((board_cnt, ROWS, COLUMNS), dtype=torch.float32)
    boards_output = torch.zeros((board_cnt, ROWS, COLUMNS), dtype=torch.float32)

    #
    # Generate 1000 random boards
    #
    for i in tqdm(range(board_cnt)):
        # Generate a random board layout
        board = empty_board(ROWS, COLUMNS)

        # This random_setup step is slow because it has to return a random _valid_
        # placement of ships with no overlap.
        random_setup(SHIP_SIZES, board)

        # Convert board to torch tensor from numpy.
        # All of this agent's code will use torch.
        board = torch.tensor(board)

        # This board represents which grid squares have a ship
        # and which are empty, example:
        # tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
        #         [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        #         [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        #         [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        #         [0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
        #         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        # Store the board as the output -- This is what our model is going to try to reconstruct.
        boards_output[i] = board

        # Now that we have our output, lets construct the representation we'll use as the models
        # input.

        # The input will represent our knowledge of the board, which is determined by how many
        # shots we've taken.

        # Choose how many shots we want for this example
        # we'll use between 0 and 99 shots
        shot_count = random.randint(0, 99)

        # Generate that many shots!
        shot_pattern = torch.rand_like(board.float()) > (shot_count / 100.0)

        # Create a representation what we know about the board state
        # We'll use a 10x10 array, where
        # 0 represents a ship
        # 1 represents a miss
        # 2 represents unknown

        # This is the representation we'll feed the neural net
        board_state = torch.zeros_like(board)
        for r in range(10):
            for c in range(10):
                if shot_pattern[r, c]:
                    # If we shot at the board in this square, we will know
                    # the board state for this square
                    if board[r, c] == Square.SHIP:
                        board_state[r, c] = 0
                    else:
                        board_state[r, c] = 1
                else:
                    # If we didn't shoot at the board, we don't know the state
                    board_state[r, c] = 2

        # Example board state:
        # tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        #         [1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        #         [2, 0, 1, 1, 2, 1, 1, 2, 1, 1],
        #         [1, 2, 1, 1, 1, 1, 1, 1, 2, 1],
        #         [1, 0, 2, 2, 1, 0, 0, 1, 2, 1],
        #         [2, 0, 2, 1, 1, 1, 0, 0, 0, 1],
        #         [1, 0, 1, 1, 0, 2, 0, 1, 1, 1],
        #         [1, 2, 1, 2, 1, 1, 2, 1, 1, 1],
        #         [1, 1, 1, 2, 2, 1, 1, 1, 1, 1]])
        boards_input[i] = board_state

    return boards_input, boards_output


def train():
    num_epochs = 500

    # We don't generate more boards because this step is slow, and memory usage
    # grows too large in our training loop.
    train_boards = 4000
    eval_boards = 100

    print("Generating training data")
    train_boards_input, train_boards_output = make_examples(train_boards)

    print("Generating eval data")
    eval_boards_input, eval_boards_output = make_examples(eval_boards)

    # Generate an optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model for num_epochs
    for i in range(num_epochs):
        optimizer.zero_grad()

        # Pass the examples through the model
        pred_boards_output = model(train_boards_input)

        # Calculate the Error between the prediction and truth
        # TODO: Should this be MSE loss or cross entropy?
        #loss = F.mse_loss(train_boards_output, pred_boards_output, reduction='mean')
        loss = F.binary_cross_entropy_with_logits(pred_boards_output, train_boards_output, reduction='mean')

        # Backprop
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            # Calculate evaluation loss
            with torch.no_grad():
                eval_loss = F.binary_cross_entropy_with_logits(model(eval_boards_input), eval_boards_output, reduction='mean')

            print(f"Epoch={i} Train Loss={loss.item():.3f} Eval Loss = {eval_loss.item():.3f}")

    print("Saving model")
    torch.save(model.state_dict(), "model.pt")


if __name__ == "__main__":
        train()

