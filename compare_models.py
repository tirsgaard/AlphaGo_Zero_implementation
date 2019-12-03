# this function is written under the assumption that model1 is the current
# best model and model2 is the competitor 

def compare_models(model1, model2, elotmp):
    import elo
    from collections import Counter


    i.addPlayer(model1, rating = elotmp)
    i.addPlayer(model2, rating = elotmp)

    
    winner = 0
    winnerList = []
        
    for mmm in range(100):
        # assuming playGame outputs the name of the model that wins
        winner = playGame(model1, model2)
        i.recordMatch(model1, model2, winner = winner)
        winnerList.append(winner)
        
    
    if Counter(winnerList)[model2] >= 55:
        elo = i.getPlayerRating(model2)
        i.removePlayer(model1)
        i.removePlayer(model2)
        return (model2, elo)
    else:
        i.removePlayer(model1)
        i.removePlayer(model2)
        return (model1, elotmp)