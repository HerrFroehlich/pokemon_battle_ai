from dataclasses import dataclass, field

from src.models.agent import Agent, AgentConfig, TeamConfig
import pokemonpython.sim.sim as sim

@dataclass
class CompetitionConfig(object):
    N_BATTLES:int = 10000
    N_BATTLES_WITH_SAME_TEAM:int = 100
    AGENTCONFIG:AgentConfig = AgentConfig()


class Competition_1vs1(object):
    
    def __init__(self, n_battles:int, conf:CompetitionConfig = CompetitionConfig()):
    
        self._conf = conf

        self._tconf = TeamConfig()
        self._tconf.N_POKEMON = 1
        self._tconf.ALLOW_ITEMS = False

        aconf = conf.AGENTCONFIG

        self._agents = [Agent(aconf, self._tconf), Agent(aconf, self._tconf)]
        self._agent_wins = [0, 0]

        self._teams = [self._agents[0].generate_team(), self._agents[1].generate_team()]


        #TODO GUI
    def add_displayable(self):
        pass



    def run(self):
        
        for battle_cnt in range(self._conf.N_BATTLES):
            if (battle_cnt % self._conf.N_BATTLES_WITH_SAME_TEAM) == 0:
                self._teams = [self._agents[0].generate_team(), self._agents[1].generate_team()]
            print("#------ self._battle %d", battle_cnt+1) # TODO RM
            self._battle = sim.new_battle('single', "Team1", self._teams[0], "Team2", self._teams[1], debug=True) # TODO debug-> false
            self._agents[0].join_battle(self._battle, 0)
            self._agents[1].join_battle(self._battle, 1)
        #   self._displayable.setbattke(self._battle)
            self._run_battle()


    def get_winner(self) -> Agent:
        pass

    def get_loser(self) -> Agent:
        pass
    
    def _run_battle(self):
        while not self._battle.ended:
            for i in range(2):
                choicestr, mv = self._agents[i].select_action()
                if mv != None:
                    choicestr += ' ' + str(mv)
                sim.choose(self._battle,i+1, choicestr)

            
            sim.do_turn(self._battle)

            for i in range(2):
                reward = self._agents[i].end_turn()
                print("Team %d : reward %0.2f" % (i, reward)) # TODO RM
                loss = self._agents[i].optimize()
                print("Team %d : loss %0.2f" % (i, loss)) # TODO RM


        if self._battle.winner == 'p1':
            self._agent_wins[0] += 1
        elif self._battle.winner == 'p2':
            self._agent_wins[1] += 1
        else :
            print("No winner this round") # TODO rm
             #self._displayable.update()

