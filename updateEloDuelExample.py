from elo import Implementation
import updateEloDuel

i = Implementation()
i.addPlayer("model1",rating=0)
i.addPlayer("model2",rating=0)
temp = updateEloDuel("model1", "model2", 62)
print(temp)