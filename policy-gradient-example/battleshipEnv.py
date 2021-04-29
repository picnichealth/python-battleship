import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np


class BattleshipEnv:
    """Battleship game environment with gym-style API
 
    Simple implementation hardwired to have two players (does n-player battleship 
    exist?). Each player has hidden and visible game boards:
    
         Hidden  -> the placement of their own ships 
         Visible -> their knowledge of the other player's ship locations represented 
           as a grid of probabilities initialized to 0.5
           
                      
    The primary API is the 'step' function, which takes an action:
    
        Action (list) := [Player_id, row, col] where the id is 0/1 according
           to the player firing the shot.
           
    step returns [new_state, reward, done] where new_state corresponds to the firing
    player's updated visible knowledge. 
 
    """    
    boards      = []    
    plt_ax      = None
    plt_fig     = None

    boardShape  = ()
    initMethods = []
    
        
    def __init__(self, init=['random', 'random'], rows=10, cols=10):

        self.boardShape  = (rows, cols)
        self.initMethods = list(init)

        for n in range(2):                    
            visible = 0.5 * np.ones( self.boardShape )
            hidden, ships  = self.initRandBoard( np.zeros( self.boardShape ), np.zeros( self.boardShape ) )
        
            self.boards.append( {"visible" : visible, "hidden" : hidden, "shipIds" : ships} )
                        
        
    def reset(self):
    
        self.boards = []
        self.__init__(init=self.initMethods, rows=self.boardShape[0], cols=self.boardShape[1])
        return [self.boards[0]["visible"], np.zeros( np.unique(self.boards[0]["shipIds"]).size - 1, dtype=np.float32)],\
               [self.boards[1]["visible"], np.zeros( np.unique(self.boards[1]["shipIds"]).size - 1, dtype=np.float32)]


    def initRandBoard(self, board, ships):
    
        shipId = 1
        # Loop over ships by 1D size:
        for ship in [5, 4, 3, 3, 2]:
        
            # Try to place ship until success        
            while (1):            
            
                # Pick a random orientation (0 -> vertical, 1 -> horizontal)
                orient = np.random.randint(2)
                            
                x1     = np.random.randint(board.shape[0])
                y1     = np.random.randint(board.shape[1])
                
                x2     = x1 + orient*ship + (1 - orient)*1            
                y2     = y1 + (1 - orient)*ship + (orient)*1
                        
                                                
                # Check bounds and that no ship already occupies this spot; place
                # and break on success. 
                if (x2 <= board.shape[0] and y2 <= board.shape[1] 
                     and board[x1:x2, y1:y2].sum() == 0.0 ):
                                          
                  board[x1:x2, y1:y2] = 1.0
                  ships[x1:x2, y1:y2] = shipId
                  
                  shipId += 1                  
                  break
                     
                                  
        return board, ships
    
    


    def step(self, action):
        
        offense_id = action[0]
        coord_r    = action[1]
        coord_c    = action[2]
        defense_id = 1 - action[0]

        if ( self.boards[offense_id]["visible"][coord_r, coord_c] !=
             self.boards[defense_id]["hidden"][coord_r, coord_c]):             

            # Fire the shot by updating player's visible state:
            self.boards[offense_id]["visible"][coord_r, coord_c] = \
                self.boards[defense_id]["hidden"][coord_r, coord_c]

            reward = self.boards[offense_id]["visible"][coord_r, coord_c]
            
        else:
            reward = -0.1
            
        # New state: updated visible state
        new_board = self.boards[offense_id]["visible"]#.ravel()
                

        # Done if we've hit all ship slots.
        if (np.sum( self.boards[offense_id]["visible"] == 1.0 ) == 17):
          done = True
          reward += 20.0
        else:
          done = False
          
          
        # Finally, we're return a bit vector of meta-info telling us 
        # which ships we've sunk, e.g. 'you sunk my battleship!'
        new_info       = np.ones( np.unique(self.boards[defense_id]["shipIds"]).size, dtype=np.float32 )
        
        shipsLeft = (self.boards[offense_id]["visible"] == 0.5) * self.boards[defense_id]["shipIds"]
        new_info[np.unique(shipsLeft).astype(np.int32)] = 0.0
        new_info = new_info[1:]
                       
        return [new_board, new_info], reward, done
    
    
    
        
    def showBoard(self, player_id=None, policy=[None, None]):
        
        if (self.plt_ax is None):
          plt.ion()          
          self.plt_fig, self.plt_ax = plt.subplots(nrows=2, ncols=4)
          self.plt_ax[1, 0].remove()
          self.plt_ax[1, 2].remove()
        
        # If no arg, show both players:
        if (player_id is None):
          self.showBoard(0, policy=policy)
          self.showBoard(1, policy=policy)

          plt.show()
          plt.pause(0.01)
          
        else:
        
            self.plt_ax[0, player_id*2].clear()
            self.plt_ax[0, player_id*2 + 1].clear()
            self.plt_ax[1, player_id*2 + 1].clear()
            
            self.plt_ax[0, player_id*2 + 1].set_title("Player " + str(player_id + 1))
            self.plt_ax[0, player_id*2 + 1].imshow(self.boards[player_id]["visible"], 
                                      cmap='coolwarm', vmin=0, vmax=1)

            self.plt_ax[0, player_id*2].set_title("AI " + str(player_id + 1))

            if (policy[player_id] is not None):
            
                choiceIdx = np.where(self.boards[player_id]["visible"] == 0.5)
            
                maxColor = np.max(policy[player_id][choiceIdx])
                minColor = np.min(policy[player_id][choiceIdx])
                minColor -= (maxColor - minColor)*0.50
                
                self.plt_ax[0, player_id*2].imshow(policy[player_id], 
                                          cmap='binary_r', vmin=minColor, vmax=maxColor)
                                          
                rgba_colors = cm.coolwarm(self.boards[player_id]["visible"])                  
                rgba_colors[:, :, 3] = (self.boards[player_id]["visible"] != 0.5).reshape(10,10)

                self.plt_ax[0, player_id*2].imshow(rgba_colors, 
                                          cmap='binary', vmin=0, vmax=1)
                
            else:
                self.plt_ax[0, player_id*2].imshow(0.5*np.ones( self.boards[player_id]["visible"].shape ), 
                                          cmap='coolwarm') #, vmin=0, vmax=1)
            

            self.plt_ax[0, player_id*2].set_xticks(np.arange(0, self.boardShape[1]));
            self.plt_ax[0, player_id*2].set_yticks(np.arange(0, self.boardShape[0]));
            self.plt_ax[0, player_id*2].set_xticklabels(np.arange(0, self.boardShape[1]));
            self.plt_ax[0, player_id*2].set_yticklabels(np.arange(0, self.boardShape[0]));
            self.plt_ax[0, player_id*2].set_xticks(np.arange(-.5, self.boardShape[1]), minor=True);
            self.plt_ax[0, player_id*2].set_yticks(np.arange(-.5, self.boardShape[0]), minor=True);
            self.plt_ax[0, player_id*2].grid(which='minor')        

                                  
            self.plt_ax[0, player_id*2 + 1].set_xticks(np.arange(0, self.boardShape[1]));
            self.plt_ax[0, player_id*2 + 1].set_yticks(np.arange(0, self.boardShape[0]));
            self.plt_ax[0, player_id*2 + 1].set_xticklabels(np.arange(0, self.boardShape[1]));
            self.plt_ax[0, player_id*2 + 1].set_yticklabels(np.arange(0, self.boardShape[0]));
            self.plt_ax[0, player_id*2 + 1].set_xticks(np.arange(-.5, self.boardShape[1]), minor=True);
            self.plt_ax[0, player_id*2 + 1].set_yticks(np.arange(-.5, self.boardShape[0]), minor=True);
            self.plt_ax[0, player_id*2 + 1].grid(which='minor')        
    
    
            self.plt_ax[1, player_id*2 + 1].imshow(self.boards[player_id]["hidden"], 
                                      cmap='coolwarm', vmin=0, vmax=1, alpha=0.6)
            self.plt_ax[1, player_id*2 + 1].set_xticklabels([])
            self.plt_ax[1, player_id*2 + 1].set_yticklabels([])    

