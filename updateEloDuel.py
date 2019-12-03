import random
from elo import Implementation

def updateEloDuel(league, player1, player2, N_wins, N_games_duel): 
    # Calculate elo using results, where player 1 is the challenger
    wins = []
    for m in range(N_wins):
        wins.append(1)
    for mm in range(N_games_duel-N_wins):
        wins.append(0)

    # Shuffle
    random.shuffle(wins)

    
    for mmm in range(N_games_duel):
        if wins[mmm] == 1:
            league.recordMatch(player1, player2, winner=player1, constant2=True)
        else:
            league.recordMatch(player1, player2, winner=player2, constant2=True)
        
    return league.getPlayerRating(player1)


vals = []
for i in range(10**5):
    league = Implementation()
    league.addPlayer("model1",rating=0)
    league.addPlayer("model2",rating=0)
    vals.append(updateEloDuel(league, "model1", "model2", 62, 100))


print(np.array(vals).mean())
