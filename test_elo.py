from elo import *
import random

i = Implementation()

i.addPlayer("A",rating=200000)
i.addPlayer("B",rating=100000)

i.recordMatch("A","B",winner="B")

print(i.getRatingList())




i.addPlayer("AlphaGo")
i.addPlayer("VoresGo",rating=900)

print(i.getPlayerRating("AlphaGo"), i.getPlayerRating("VoresGo"))

i.recordMatch("AlphaGo","VoresGo",winner="AlphaGo")

print(i.getRatingList())

i.recordMatch("AlphaGo","VoresGo",winner="VoresGo")

print(i.getRatingList())

i.recordMatch("AlphaGo","VoresGo",draw=True)

print(i.getRatingList())

i.removePlayer("AlphaGo")

print(i.getRatingList())


i = Implementation()

i.addPlayer("AlphaGo")
i.addPlayer("VoresGo",rating=900)

# wins = np.random.random(100)
random.shuffle(wins)

for i in range(100):
    if wins[1] > 0.5:
        i.recordMatch("AlphaGo", "VoresGo", winner="VoresGo", constant1=True)


print(i.getRatingList())
    


"""
import time

start = time.time()
for x in range(10**6):
    i.recordMatch("AlphaGo", "VoresGo", winner="VoresGo")
end = time.time()
print("Time for one million elo simulations: ", end - start, "seconds")
print(i.getRatingList())

i.addPlayer("AlphaGo")
i.addPlayer("VoresGo")

for x in range(45):
    i.recordMatch("AlphaGo", "VoresGo", winner="VoresGo")
    
for x in range(55):
    i.recordMatch("AlphaGo", "VoresGo", winner="AlphaGo", constant1=True)
print(i.getRatingList())
"""