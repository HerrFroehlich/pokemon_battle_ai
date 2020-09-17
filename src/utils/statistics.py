import pandas as pd
import numpy as np
import os
import math

from pokemonpython.sim.structs import Pokemon, Battle, get_active_pokemon
import pokemonpython.data.dex as dex


    
_type_effectivity_table =  pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/type_effictivity.csv'), index_col=0)
_move_type_table = pd.DataFrame().from_dict(dex.move_dex) #TODO use

def calc_modifier(mvname, user, target):
    # mvname = user.moves[mv_idx]
    move = dex.move_dex[mvname]
    if move.category == 'Status':
        return float('NaN')

    modifier = 1
    # STAB (same type attack bonus)
    if move.type in user.types:
        modifier *= 1.5

    # TYPE EFFECTIVENESS
    for each in target.types:
        type_effect = _type_effectivity_table.at[move.type,each]
        modifier *= type_effect

    return modifier

def _calc_running_avg(old_avg:float, x:float, N:int):
    return (old_avg + (x-old_avg)/N)

class IAgentStats(object):
    mv_cnts = [0,0,0,0]
    n_random = 0
    n_passed = 0
    avg_reward = 0.0
    avg_loss = 0.0
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

    def __str__(self):
        return '''\
#------------ AGENT STATISTIC ------------#
# No statistics where recorded
#-----------------------------------------# \n'''

class AgentStats(IAgentStats):

    def __init__(self):
        super(AgentStats, self).__init__()
        self._n_reward = 0
        self._n_loss = 0
        self.mv_cnts = [0,0,0,0]
        self.n_random = 0
        self.n_passed = 0
        self.avg_reward = 0.0
        self.avg_loss = 0.0

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
        self._n_reward = 0
        self._n_loss = 0
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


class IBattleStats(object):
    def __init__(self, stat_team1:IAgentStats, stat_team2:IAgentStats, log_step:int = 1):
        pass

    def eval_battle(self, battle:Battle):
        pass

    def __str__(self):
        pass

class BattleStatsStub(IBattleStats):
    
    def __init__(self):
        pass

    def __init__(self, stat_team1:IAgentStats, stat_team2:IAgentStats, log_step:int = 1):
        pass

    def __str__(self):
        return '''\
#------------ BATTLE STATISTIC ------------#
# No statistics where recorded
#-----------------------------------------# \n'''

class BattleStats(IBattleStats):
    columns = ["TEAM1", "TEAM2"]
    rows = ["effective_cnt", "ineffective_cnt", "normal_cnt", "status_move_cnt", "avg_reward", "avg_loss"]
    
    effective_range = [2, 6.0] #inclusive range
    normal_range = [1.0, 1.5] #inclusive range
    ineffective_range = [0, 0.75] #inclusive range


    def __init__(self, stat_team1:IAgentStats, stat_team2:IAgentStats, log_step:int = 1):
        
        self.timestamps = np.array([], dtype=int)
        self.effective_cnt = np.empty((0,2), dtype=int)
        self.ineffective_cnt = np.empty((0,2), dtype=int)
        self.normal_cnt = np.empty((0,2), dtype=int)
        self.status_move_cnt = np.empty((0,2), dtype=int)
        self.random_cnt = np.empty((0,2), dtype=int)
        self.avg_reward = np.empty((0,2), dtype=float)
        self.avg_loss = np.empty((0,2), dtype=float)
        self.battles_won_cnt = np.empty((0,2), dtype=int)
        
        
        self._n_battles = 0
        self._LOG_STEP = log_step
        self._team1_stat = stat_team1
        self._team2_stat = stat_team2

        self._current_effective_cnt = np.zeros((1,2), dtype=int)
        self._current_ineffective_cnt = np.zeros((1,2), dtype=int)
        self._current_normal_cnt = np.zeros((1,2), dtype=int)
        self._current_status_move_cnt = np.zeros((1,2), dtype=int)
        self._current_battles_won_cnt = np.zeros((1,2), dtype=int)
        
        
    
    def eval_battle(self, battle:Battle):
        self._n_battles += 1
        pkms = get_active_pokemon(battle) #TODO double fights

        if battle.winner == 'p1':
            self._current_battles_won_cnt[0,0] += 1
        elif battle.winner == 'p2':
            self._current_battles_won_cnt[0,1] += 1


        team1_effectivies = [calc_modifier(mv,  pkms[0], pkms[1]) for mv in pkms[0].moves]
        team2_effectivies = [calc_modifier(mv, pkms[1],  pkms[0]) for mv in pkms[1].moves]

        self._extract_mv_data(0, self._team1_stat, team1_effectivies)
        self._extract_mv_data(1, self._team2_stat, team2_effectivies)

        if (self._n_battles % self._LOG_STEP) == 0:
            self.effective_cnt = np.append(self.effective_cnt, self._current_effective_cnt, axis = 0)
            self.ineffective_cnt = np.append(self.ineffective_cnt, self._current_ineffective_cnt, axis = 0)
            self.normal_cnt = np.append(self.normal_cnt, self._current_normal_cnt, axis = 0)
            self.status_move_cnt = np.append(self.status_move_cnt, self._current_status_move_cnt, axis = 0)
            self.battles_won_cnt = np.append(self.battles_won_cnt, self._current_battles_won_cnt, axis = 0)

            self.random_cnt = np.append(self.random_cnt, [[self._team1_stat.n_random, self._team2_stat.n_random]], axis = 0)
            
            self.avg_reward = np.append(self.avg_reward, [[self._team1_stat.avg_reward, self._team2_stat.avg_reward]], axis = 0)
            self.avg_loss = np.append(self.avg_loss, [[self._team1_stat.avg_loss, self._team2_stat.avg_loss]], axis = 0)
            self.timestamps = np.append(self.timestamps,self._n_battles)

            self._current_effective_cnt = np.zeros((1,2), dtype=int)
            self._current_ineffective_cnt = np.zeros((1,2), dtype=int)
            self._current_normal_cnt = np.zeros((1,2), dtype=int)
            self._current_status_move_cnt = np.zeros((1,2), dtype=int)
            self._current_battles_won_cnt = np.zeros((1,2), dtype=int)
            self._team1_stat.reset()
            self._team2_stat.reset()


    def _extract_mv_data(self, team_col, team_stat:IAgentStats, team_effectivies):
        for mv_idx in range(len(team_effectivies)):
            mult = team_effectivies[mv_idx]
            if math.isnan(mult):
                self._current_status_move_cnt[0,team_col] += team_stat.mv_cnts[mv_idx]
                team_stat.mv_cnts[mv_idx] = 0
            elif BattleStats.effective_range[0] <= mult <= BattleStats.effective_range[1]:
                self._current_effective_cnt[0,team_col] += team_stat.mv_cnts[mv_idx]
                team_stat.mv_cnts[mv_idx] = 0
            elif BattleStats.ineffective_range[0] <= mult <= BattleStats.ineffective_range[1]:
                self._current_ineffective_cnt[0,team_col] += team_stat.mv_cnts[mv_idx]
                team_stat.mv_cnts[mv_idx] = 0
            elif BattleStats.normal_range[0] <= mult <= BattleStats.normal_range[1]:
                self._current_normal_cnt[0,team_col] += team_stat.mv_cnts[mv_idx]
                team_stat.mv_cnts[mv_idx] = 0
            else:
               print("ERROR: Invalid Range for mult %f", mult)

    def __str__(self):
        string = '''\
#------------ BATTLE STATISTIC ------------#
# Results after %d battles:\n
#           [TEAM1 TEAM2]
# Effective:   %s
# Ineffective: %s
# Normal:      %s
# Status moves:%s
# Avg rewards: %s
# Avg loss:    %s
#-----------------------------------------# \n''' % (self._n_battles, np.sum(self.effective_cnt, axis=0), np.sum(self.ineffective_cnt, axis=0),\
                      np.sum(self.normal_cnt, axis=0), np.sum(self.status_move_cnt, axis=0), np.average(self.avg_reward, axis=0), np.average(self.avg_loss, axis=0))

        return string
