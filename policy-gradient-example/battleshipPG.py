"""
Policy Gradient RL for battleship.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import pdb

import battleshipEnv as bse        

# ======================================================================================
#
# Params (game & AI training)
#
# ======================================================================================

boardR = 10     # Board rows/cols
boardC = 10

epochs = 5000
reward_discount_gamma = 0.9


# ======================================================================================
#
# Policy network definition:
#
# ======================================================================================
class Policy(nn.Module):

        """Two layer network; input is board & a bit vector indicating which
        ships have been already been sunk; process the board at layer 1, 
        then concatenate its activations with the ship vector as inputs two
        layer 2. 
        
        Network outputs probability of any (hidden) board position
        containing a ship. 
        
        Implements two ways calls to the network: 
        
        1) Sample a board square to fire at probabilistically (this is needed
        for policy gradient training)
        
        2) Select the highest probability square to play (this is used for
        playing games against an adversary)
        
        """        
        def __init__(self, boardR, boardC, numShips):        
                super(Policy, self).__init__()
                
                self.l1 = nn.Linear(boardR * boardC, 400)
                self.l2 = nn.Linear(400 + numShips, boardR * boardC)
                
                self.boardR   = boardR
                self.boardC   = boardC
                self.numShips = numShips

                
        def forward(self, board_vec, ship_vec):
                    
            self.h1     = F.relu(self.l1(board_vec))
            self.h1c    = torch.cat( (self.h1, ship_vec), dim=1)
            self.probs  = F.softmax(self.l2(self.h1c), dim=1)                               

            return self.probs

        
        """
        Run the foward pass, sample from the last layer, and 
        get the associated log prob of the chosen action as
        a differentiatiable variable (for training)
        """        
        def sampleAction(self, state):
            
            # Grab board and ship data from current state; reshape as needed:
            board_vec   = torch.from_numpy(state[0]).view(-1, self.boardR * self.boardC).float()
            ship_vec    = torch.from_numpy(state[1][np.newaxis, :]).float()
        
            # Run network forward.        
            actDistr = torch.distributions.Categorical( self.forward(board_vec, ship_vec) )
               
            # Sample action:
            action   = actDistr.sample()
            log_prob = actDistr.log_prob(action)
            
            return  action, log_prob
            

        """
        Run the foward pass and take the highest-probability move
        that hasn't been played yet (for playing)
        """        
        def playAction(self, state):

            board_vec   = torch.from_numpy(state[0]).view(-1, self.boardR * self.boardC).float()
            ship_vec    = torch.from_numpy(state[1][np.newaxis, :]).float()
        
            # Run network forward.
            actDistr = torch.distributions.Categorical( self.forward(board_vec, ship_vec) )
             
            # Highest prob square, with rejection sampling:   
            candidates = actDistr.probs.argsort()
            for i in reversed(range(candidates.size()[1])):
              action = candidates[0, i]
              if (board_vec[0, action] == 0.5):     # 0.5 in board_vec denotes 'hidden' square
                break
               
            # Ok, good action; let's compute log prob:   
            log_prob = actDistr.log_prob(action)
            
            return  action, log_prob


# ======================================================================================
#
# Play a one sided game; use the 'show' argument to visualize step by step. 
#
# ======================================================================================
def playOneGame(env, policy, t_max=50, show=False, play=False):

    p1_s, p2_s = env.reset()
    
    states       = []
    actions      = []
    rewards      = []     
    total_reward = 0
    moves        = 0    
    done         = False

    for t in range(t_max):
        
            # If in play mode, take MAP action; otherwise in training mode, so sample:
            if (show or play):
                action, log_prob_action = policy.playAction(p1_s)            
            else: 
                action, log_prob_action = policy.sampleAction(p1_s)            
                
    
            new_s, reward, done  = env.step([0, int(action/boardC), action%boardC])

            states.append( np.array(p1_s) )
            actions.append(log_prob_action)
            rewards.append(reward)
            
            total_reward += reward  
            moves        += 1          
            p1_s = new_s
            
            if (show):                            
              env.showBoard(policy=[policy.probs.detach().numpy().reshape(10,10), None])
              print(p1_s[1])
            
            if (done):        
              break
                             
                                    
    # Game over; compute discounted future reward:
    c = 0.0
    for i in reversed(range(len(rewards))):
      c = reward_discount_gamma * c + rewards[i]
      rewards[i] = c
      
    # 'Baseline' technique:
    rewards = np.array(rewards)
    rewards -= np.mean(rewards)
    rewards /= np.std(rewards)


    if (show):
        plt.close()
        env.plt_ax = None
        env.plt_fig = None

    return actions, states, rewards, total_reward, moves






def playNGames(env, pol, n):

    batchA, batchS, batchR, batchTr, batchM = [], [], [], [], []

    for i in tqdm(range(n)):
      a, s, r, tr, m = playOneGame(env, pol, t_max=100)
      
      batchA.extend(a)
      batchS.extend(s)
      batchR.extend(r)
      batchTr.append(tr) 
      batchM.append(m)
                  
    return batchA, batchS, batchR, batchTr, batchM
    
        
                        
                        
# ======================================================================================
#
# Main training loop:
#
# ======================================================================================                        
env = bse.BattleshipEnv(rows=boardR, cols=boardC)
pol = Policy(boardR, boardC, numShips=5)
opt = optim.Adam(pol.parameters(), lr=0.00025)


# Plot loss and mean score over time:
plt.ion()
score_fig, score_axes = plt.subplots(2, 1)
meanScores            = []
losses                = []


# Just see a few games with an AI that trained for about 24 hours; then reinit
# and start training from scratch:
pol = torch.load('pgPolicy_wGamePieceFlags.tc') 
playOneGame(env, pol, show=True)
playOneGame(env, pol, show=True)
playOneGame(env, pol, show=True)

pol = Policy(boardR, boardC, numShips=5)


for epoch in range(epochs):

    batchActions, batchStates, batchRewards, batchTotRewards, batchMoves = \
        playNGames(env, pol, 64)


    loss = Variable( torch.zeros(size=(1,1)) )

    # Loss is negative log probability * reward for taking an action (over
    # all moves we've played):
    for i in range(len(batchActions)):
        loss += -batchActions[i] * batchRewards[i]

    opt.zero_grad()
    loss.backward()
    opt.step()

    # Periodically visualize how the AI plays through a game
    if (epoch % 20 == 0):
        playOneGame(env, pol, show=True)


    # Plot average score and loss:
    meanScores.append( np.mean(batchTotRewards) )
    losses.append( loss.detach().numpy()[0,0] )
    
    score_axes[0].clear()
    score_axes[0].plot(meanScores)

    score_axes[1].clear()
    score_axes[1].plot(losses)

    plt.pause(0.01)

    