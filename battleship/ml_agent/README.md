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
Here is the output from training the model 1000 steps at a learning rate of 0.001:

- Epoch=0     Train Loss = 0.693   Eval Loss = 0.660
- Epoch=100   Train Loss = 0.305   Eval Loss = 0.347
- Epoch=200   Train Loss = 0.162   Eval Loss = 0.287
- Epoch=300   Train Loss = 0.089   Eval Loss = 0.375
- Epoch=400   Train Loss = 0.031   Eval Loss = 0.605
- Epoch=500   Train Loss = 0.016   Eval Loss = 0.768
- Epoch=600   Train Loss = 0.009   Eval Loss = 0.967
- Epoch=700   Train Loss = 0.007   Eval Loss = 1.125
- Epoch=800   Train Loss = 0.005   Eval Loss = 1.252
- Epoch=900   Train Loss = 0.005   Eval Loss = 1.355


# Evaluation
The model works!  I wish it worked better, but at least it does substantially better
 than random guessing.  Using the evaluation code (`eval.py`) I measured the number of
 hits it takes to sink a random opponent board:
 
- Average number of turns to sink random board using Random: 95.3
- Average number of turns to sink random board using ML Agent: 73.47
