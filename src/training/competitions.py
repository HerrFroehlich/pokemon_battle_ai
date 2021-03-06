from dataclasses import dataclass, field

from src.models.agent import Agent, AgentConfig, TeamConfig, IAgent, RandomAgent
from src.utils.statistics import BattleStatsStub, BattleStats, AgentStats
from src.visuals.displayable import DisplayableStub, IDisplayable

import pokemonpython.sim.sim as sim

@dataclass
class CompetitionConfig(object):
    N_BATTLES:int = 10000
    N_BATTLES_WITH_SAME_TEAM:int = 100
    ENABLE_STATS:bool = True
    STATS_LOG_STEP:int = 100
    DEBUG:bool = False
    VS_RANDOM = False
    AGENTCONFIG:AgentConfig = AgentConfig()


class Competition_1vs1(object):
    
    MAX_TURNS_WITH_BOTH_PASSING = 10

    def __init__(self, conf:CompetitionConfig = CompetitionConfig()):
    
        self._conf = conf

        self._tconf = TeamConfig()
        self._tconf.N_POKEMON = 1
        self._tconf.ALLOW_ITEMS = False

        aconf = conf.AGENTCONFIG

        if conf.VS_RANDOM:
            agent2_t = RandomAgent
        else:
            agent2_t = Agent

        self._agents = [Agent(aconf, self._tconf), agent2_t(aconf, self._tconf)]
        self._agent_wins = [0, 0]

        if conf.ENABLE_STATS:
            astat1 = AgentStats()
            astat2 = AgentStats()
            self._agents[0].set_statistic_grabber(astat1)
            self._agents[1].set_statistic_grabber(astat2)
            self._stats = BattleStats(astat1, astat2, conf.STATS_LOG_STEP)
        else:
            self._stats = BattleStatsStub()


        self._teams = [self._agents[0].generate_team(), self._agents[1].generate_team()]

        self._displ = DisplayableStub()

    def add_displayable(self, displ:IDisplayable):
        self._displ = displ
    
    def get_stats(self):
        return  self._stats

    def get_winner(self) -> Agent:
        agentIdx = self._agent_wins.index(max(self._agent_wins))
        print ("#- TEAM %d won the 1vs1 competition!" % (agentIdx+1))
        return self._agents[agentIdx]

    def get_loser(self) -> Agent:
        agentIdx = self._agent_wins.index(min(self._agent_wins))
        print ("#- TEAM %d lost the 1vs1 competition!" % (agentIdx+1))
        return self._agents[agentIdx]

    def run(self):

        self._displ.initialize()
        
        for battle_cnt in range(self._conf.N_BATTLES):
            if (battle_cnt % self._conf.N_BATTLES_WITH_SAME_TEAM) == 0:
                self._teams = [self._agents[0].generate_team(), self._agents[1].generate_team()]
            print("#------ BATTLE %d" % (battle_cnt+1))
            self._battle = sim.new_battle('single', "Team1", self._teams[0], "Team2", self._teams[1], debug=self._conf.DEBUG)
            self._displ.set_battle(self._battle)
            self._battle.weather = '' #normal weather
            self._agents[0].join_battle(self._battle, 0)
            self._agents[1].join_battle(self._battle, 1)
        #   self._displayable.setbattke(self._battle)
            self._run_battle()
            self._stats.eval_battle(self._battle)


    
    def _run_battle(self):
        both_passing_cnt = 0
        while not self._battle.ended:
            have_both_passed = True
            both_mv_idx = [0, 0]
            for i in range(2):
                choicestr, mv = self._agents[i].select_action()
                both_mv_idx[i] = mv
                if mv != None:
                    choicestr += ' ' + str(mv)
                sim.choose(self._battle,i+1, choicestr)
                have_both_passed = have_both_passed and (choicestr == "pass")

            self._displ.start_turn(both_mv_idx[0], both_mv_idx[1])
            both_passing_cnt += have_both_passed
            if (both_passing_cnt > Competition_1vs1.MAX_TURNS_WITH_BOTH_PASSING):
                print("Both passing for to long -> ending battle")
                break

            sim.do_turn(self._battle)

            for i in range(2):
                reward = self._agents[i].end_turn()
                # print("Team %d : reward %0.2f" % (i, reward)) # TODO RM
                loss = self._agents[i].optimize()
                # print("Team %d : loss %0.2f" % (i, loss)) # TODO RM
            self._displ.end_turn()

        if self._battle.winner == 'p1':
            self._agent_wins[0] += 1
        if self._battle.winner == 'p2':
            self._agent_wins[1] += 1
             #self._displayable.update()

