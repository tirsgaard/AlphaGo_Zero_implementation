import time
import numpy as np
from go import go_board

player_turn = []
for i in range(70):
    if ((i % 2)==0):
        player_turn.append("black")
    else:
        player_turn.append("white")

# ***** Choices *****#
num_games = 1000
num_rounds = 70
MCTS_sim = 400


start_eval_game_time = 0
num_iter_time = 0
t = time.time()
np.random.seed(10)
for i in range(num_games):
    t2 = time.time()
    game = go_board(9)
    start_eval_game_time += time.time()-t2 
    
    t3 = time.time()
    for j in range(num_rounds):
        possible_board = game.get_legal_board(player_turn[j])
        possible_board = np.reshape(possible_board, (81))
        possible_board = possible_board/np.sum(possible_board)
        choice = np.random.choice(81,1,list(possible_board))[0]
        choice = np.unravel_index(choice, (9,9))
        game.move(choice, player_turn[j])
        
    num_iter_time += time.time()-t3
    
    t2 = time.time()
    game.count_points()
    start_eval_game_time += time.time()-t2
        

actual_runtime = time.time()-t-num_games*num_rounds*2.3224377632141113e-05


print("Total amount of games pr. week: ", (num_games/actual_runtime)*3600*24*7)

choice_time = 70*6.010079383850098e-06
realistic_runtime = num_games/(start_eval_game_time+MCTS_sim*(num_iter_time-choice_time))

print("Total amount of MCTS games pr. week: ", realistic_runtime*3600*24*7)