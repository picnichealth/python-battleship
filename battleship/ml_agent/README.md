# ML Battleship Agent
This is my ML-based agent for playing battleship!


# Approach
The approach here is to use a CNN to help us play battleship.  The goal of the model is to
answer the question:  Given the known state of the board -- specifically which squares are
hit, missed, or unknown -- what is our best estimate for the probability that an unknown
square is occupied by a ship.

For instance, if there is a sequence of \[Miss, Hit, Unknown\] the probability is high
that the unknown square has a ship.  The model should be able to learn this.

If our model can estimate this conditional probability, P(square occupied | board state),
we can simply repeatedly fire on the spots that our model thinks are most
likely to contain a ship!


# Training
Here is the output from training the model 500 steps at a learning rate of 0.001:

- Epoch=0    Train Loss=0.693  Eval Loss = 0.661
- Epoch=100  Train Loss=0.325  Eval Loss = 0.341
- Epoch=200  Train Loss=0.206  Eval Loss = 0.225
- Epoch=300  Train Loss=0.165  Eval Loss = 0.199
- Epoch=400  Train Loss=0.148  Eval Loss = 0.202


# Evaluation
The model works!  I wish it worked better, but at least it does substantially better
 than random guessing.  Using the evaluation code (`eval.py`) I measured the number of
 hits it takes to sink a random opponent board:
 
- Average number of turns to sink random board using Random: 95.3
- Average number of turns to sink random board using ML Agent: 64.5
