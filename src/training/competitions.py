from src.models.agent import Agent, AgentConfig, TeamConfig
import pokemonpython.sim.sim as sim

class Competition_1vs1(object):
    
    def __init__(self, aconf:AgentConfig):
        self._tconf = TeamConfig()
        self._tconf.N_POKEMON = 1
        self._tconf.ALLOW_ITEMS = False

        self._agent1 = Agent(aconf, tconf)
        self._agent2 = Agent(aconf, tconf)

        self._battle = new_battle('single', "Team1", self._agent1.generate_team(), "Team2", self._agent2.generate_team(), debug=True)
        
        #TODO GUI
        #self._displayable = Displayable(self._battle)

    def run(self):
        while not self._battle.ended:
            ## do turn
        #self._displayable.update()
        