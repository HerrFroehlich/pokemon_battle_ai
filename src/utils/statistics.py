import pandas as pd


def _calc_running_avg(old_avg:float, x:float, N:int):
    return (old_avg + (x-old_avg)/N)

class IAgentStats(object):
    def __init__(self):
        pass
    def log_move(self, move:int, was_rand:bool):
        pass
    def log_reward(self, reward:float):
        pass
    def log_loss(self, reward:float):
        pass
    def reset(self):
        pass

    def __str__(self):
        pass

class AgentStatsStub(IAgentStats):
    def __init__(self):
        super(AgentStatsStub, self).__init__()

class AgentStats(IAgentStats):
    mv_cnts = [0,0,0,0]
    n_random = 0
    n_passed = 0
    avg_reward = 0.0
    avg_loss = 0.0

    def __init__(self):
        super(AgentStats, self).__init__()
        self._n_reward = 0
        self._n_loss = 0

    def log_move(self, move:int, was_rand:bool):
        if (move == None):
            self.n_passed +=1
        else:
            self.n_random += was_rand  # converts to int
            self.mv_cnts[move] += 1
    
    def log_reward(self, reward:float):
        self._n_reward += 1
        self.avg_reward = _calc_running_avg(self.avg_reward, reward, self._n_reward)

    def log_loss(self, loss:float):
        self._n_loss += 1
        self.avg_loss = _calc_running_avg(self.avg_loss, loss, self._n_loss)
    
    def reset(self):
        self.mv_cnts = [0,0,0,0]
        self.n_random = 0
        self.n_passed = 0
        self.avg_reward = 0.0
        self.avg_loss = 0.0

    def __str__(self):
        str =  '''\
#------------ AGENT STATISTIC ------------#
# Move 0 Cnt: %d
# Move 1 Cnt: %d
# Move 2 Cnt: %d
# Move 3 Cnt: %d
# Nof random choices: %d
# Nof times passed: %d
# Avg. reward: %.2f
# Avg. loss: %.2f
#-----------------------------------------# \n''' %\
        (self.mv_cnts[0], self.mv_cnts[1], self.mv_cnts[2], self.mv_cnts[3],\
        self.n_random, self.n_passed, self.avg_reward, self.avg_loss)
        return str
