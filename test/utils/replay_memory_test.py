from src.utils.replay_memory import ReplayMemory

r = ReplayMemory(3)
r.push(0,1,2,3)
r.push(3,4,5,6)
r.push(10,9,8,7)
print (len(r))
print (r.sample(3))
print (r.sample(2))
print (r.sample(1))

r.push(10,9,8,7)
print (len(r))
print (r.sample(3))