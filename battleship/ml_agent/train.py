from torch import optim, nn
from torch.nn import functional as F

from battleship.board import empty_board
from battleship.bot import random_setup
from tqdm import tqdm
import torch
import random
from battleship.ml_agent.model import model

ROWS = 10
COLUMNS = 10

SHIP_SIZES = [
    5,  # carrier
    4,  # battleship
    3,  # cruiser
    3,  # submarine
    2,  # destroyer
]

def make_examples():
    boards_input = torch.zeros((1000, 10, 10), dtype=torch.float32)
    boards_output = torch.zeros((1000, 10, 10), dtype=torch.float32)

    #
    # Generate 1000 random boards
    #
    for i in tqdm(range(1000)):
        # Generate a random board layout
        board = empty_board(ROWS, COLUMNS)
        random_setup(SHIP_SIZES, board)

        # Convert board to torch tensor from numpy.
        # All of this agents code will use torch.
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

        # Store the board as the output
        # This is what our model is going to try to reconstruct
        boards_output[i] = board

        # Now that we have our output, lets construct
        # the representation we'll use as the models input.

        # The input will represent our knowledge of the board,
        # which is determined by how many shots we've taken.

        # Choose how many shots we want for this example
        # we'll use between 0 and 99 shots
        shot_count = random.randint(0, 99)

        # Generate that many shots!
        shot_pattern = torch.rand_like(board.float()) > (shot_count / 100.0)

        # Create a representation what we know about the board state
        # We'll use a 10x10 array, where
        # 0 represents a miss
        # 1 represents a hit
        # -1 represents unknown
        # This is the representation we'll feed the neural net
        board_state = torch.where(shot_pattern,
                board,
                -1 * torch.ones_like(shot_pattern))

        # Example board state:
        # tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #         [0., -1., 1., 1., -1., 1., 1., 0., 0., -1.],
        #         [0., 1., 0., 0., -1., 0., 0., 0., 0., 0.],
        #         [0., -1., 0., 0., 1., 0., 0., 0., 0., 0.],
        #         [0., 1., 0., -1., 1., 0., 0., 0., 0., 0.],
        #         [0., 1., 0., -1., 0., 0., -1., 0., -1., 0.],
        #         [-1., 0., -1., 0., 0., 0., 0., 0., 0., 0.],
        #         [0., -1., 0., 1., 1., 1., 0., 0., 0., 0.],
        #         [-1., 0., -1., 0., -1., 0., 0., -1., -1., 0.],
        #         [0., 0., -1., 0., 0., 1., 1., 0., 0., 0.]])
        boards_input[i] = board_state

    return boards_input, boards_output


print("Generating training data")
train_boards_input, train_boards_output = make_examples()
print("Generating eval data")
eval_boards_input, eval_boards_output = make_examples()


def train(model):
    # Generate an optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # Train the model for 100 epochs
    for i in range(500):
        optimizer.zero_grad()

        # Pass the examples through the model
        pred_boards_output = model(train_boards_input)

        # Calculate the RMS error between the prediction and truth
        #loss = F.mse_loss(train_boards_output, pred_boards_output, reduction='mean')
        loss = F.binary_cross_entropy_with_logits(pred_boards_output, train_boards_output, reduction='mean')

        # Backprop
        loss.backward()

        optimizer.step()

        if i % 10 == 0:
            # Calculate evaluation loss
            with torch.no_grad():
                eval_loss = F.binary_cross_entropy_with_logits(model(eval_boards_input), eval_boards_output, reduction='mean')

            print(f"Step={i} Train Loss={loss.item():.3f} Eval Loss = {eval_loss.item():.3f}")


train(model)

print("Saving model")
torch.save(model.state_dict(), "model.pt")

