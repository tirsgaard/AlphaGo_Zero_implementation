"""
Implementation of elo. 

It is a list of players with an elo attribute
"""

class Implementation:
    """
    a class that represents an implementation of the elo System
    """

    def __init__(self, base_rating=1500) :
        # runs at initialization of class object. base_rating is the default 
        # elo a new player would have. 
        
        self.base_rating = base_rating
        self.players = []

    def __getPlayerList(self):

        # returns a list of players
        
        return self.players

    def getPlayer(self, name):

        # finds a player with the given name
        
        for player in self.players:
            if player.name == name:
                return player
        return None

    def contains(self, name):

        # returns true if there exists a player with that name, false otherwise
        # mostly used for debugging
        
        for player in self.players:
            if player.name == name:
                return True
        return False

    def addPlayer(self, name, rating=None):
        
        # adds a player to the list of players
        
        if rating == None:
            rating = self.base_rating

        self.players.append(_Player(name=name,rating=rating))

    def removePlayer(self, name):

        # removes a player from the list of players
        
        self.__getPlayerList().remove(self.getPlayer(name))


    def recordMatch(self, name1, name2, winner=None, draw=False, constant1=False, constant2=False):

        # we should call this after a game is decided. see test_elo for
        # useage of this
        
        # we can modify k as we see fit. possibly experiment?
        
        # implemented draw in case we want to use it later
        
        player1 = self.getPlayer(name1)
        player2 = self.getPlayer(name2)

        expected1 = player1.compareRating(player2)
        expected2 = player2.compareRating(player1)
        
        k = 20

        rating1 = player1.rating
        rating2 = player2.rating

        if draw:
            score1 = 0.5
            score2 = 0.5
        elif winner == name1:
            score1 = 1.0
            score2 = 0.0
        elif winner == name2:
            score1 = 0.0
            score2 = 1.0
        else:
            raise ValueError('One of the names must be the winner or draw must be True')
        
        if constant1:
            newRating1 = rating1
            newRating2 = rating2 + k * (score2 - expected2)
        elif constant2: 
            newRating2 = rating2
            newRating1 = rating1 + k * (score1 - expected1)
        else:
            newRating1 = rating1 + k * (score1 - expected1)
            newRating2 = rating2 + k * (score2 - expected2)

        # avoids negative elos

    #    if newRating1 < 0:
    #        newRating1 = 0
    #        newRating2 = rating2 - rating1

    #    elif newRating2 < 0:
    #        newRating2 = 0
    #        newRating1 = rating1 - rating2

        player1.rating = newRating1
        player2.rating = newRating2

    def getPlayerRating(self, name):

        # returns the elo of a player with name
        
        player = self.getPlayer(name)
        return player.rating

    def getRatingList(self):
 
        # returns a list of tuples in the form of ({name},{rating})


        lst = []
        for player in self.__getPlayerList():
            lst.append((player.name,player.rating))
        return lst

class _Player:

    # a class to represent players in our elo system

    def __init__(self, name, rating):

        # initializes a player with the name name and elo rating
        
        self.name = name
        self.rating = rating

    def compareRating(self, opponent):

        # compares the elos of two players and returns the expected score
        
        return ( 1+10**( ( opponent.rating-self.rating )/400.0 ) ) ** -1
