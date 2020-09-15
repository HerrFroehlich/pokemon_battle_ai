from src.models.agent import Agent, AgentConfig, TeamConfig
import pokemonpython.sim.sim as sim

class Competition_1vs1(object):
    
    def __init__(self, n_battles:int, aconf:AgentConfig, enable_stats:bool = True):
        self._tconf = TeamConfig()
        self._tconf.N_POKEMON = 1
        self._tconf.ALLOW_ITEMS = False

        self._n_battles = n_battles
        self._enable_stats = enable_stats

        self._agents = [Agent(aconf, self._tconf), Agent(aconf, self._tconf)]
        self._agent_wins = [0, 0]

        
        #TODO GUI

    def add_displayable():
        pass

    def run(self):
        
        for battle_cnt in range(self._n_battles):
            print("#------ self._battle %d", battle_cnt+1) # TODO RM
            self._battle = sim.new_battle('single', "Team1", self._agents[0].generate_team(), "Team2", self._agents[1].generate_team(), debug=True) # TODO debug-> false
            self._agents[0].join_battle(self._battle, 0)
            self._agents[1].join_battle(self._battle, 1)
        #   self._displayable.setbattke(self._battle)
            self._run_battle()

    def get_winner() -> Agent:
        pass
    
    def _run_battle(self):
        while not self._battle.ended:
            for i in range(2):
                actionstr, mv = self._agents[i].select_action()
                choicestr = actionstr + ' ' + str(mv)
                sim.choose(self._battle,i+1, choicestr)

            
            sim.do_turn(self._battle)

            for i in range(2):
                reward = self._agents[i].end_turn()
                print("Team %d : reward %0.2f" % (i, reward)) # TODO RM
                self._agents[i].optimize()


        if self._battle.winner == 'p1':
            self._agent_wins[0] += 1
        elif self._battle.winner == 'p2':
            self._agent_wins[1] += 1
        else :
            print("No winner this round") # TODO rm
             #self._displayable.update()

