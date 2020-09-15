from dataclasses import dataclass
import torch
import random
import math

import pokemonpython.sim.sim as pp_sim
import pokemonpython.tools.pick_six as pp_pick_six

from src.utils.get_torch_device import GLOBAL_TORCH_DEVICE
from src.utils.replay_memory import ReplayMemory
from src.utils.states import BattleState, PokemonState
from dqn_network import SelectiveDQNConfig, SelectiveDQN
from rewards import reward_hp_diff

@dataclass
class AgentConfig(object):
    BATCH_SIZE = 128 # NOF batches used for optimizing
    EPS_START = 0.9 # epsilon start for expontential random decay function
    EPS_END = 0.05# epsilon end for expontential random decay function
    EPS_DECAY = 200 # decay gradient for expontential random decay function
    GAMMA = 0.999
    TARGET_UPDATE = 10
    MAX_MOVES = 4
    NETWORK_CONFIG = SelectiveDQNConfig()
    REWARD_FNCT = reward_hp_diff

@dataclass
class TeamConfig(object):
    N_POKEMON = 1
    # TODO level sum
    ALLOW_ITEMS = False


class Agent(object):
    def __init__(self, config:AgentConfig = AgentConfig(), teamconfig:TeamConfig = TeamConfig()):
        self._steps_done = 0
        self._config = config
        self._teamconfig = teamconfig

        #fill outt the newtork config
        self._config.NETWORK_CONFIG.D_IN = BattleState.get_tensor_size()
        self._config.NETWORK_CONFIG.D_OUT = PokemonState.MAX_MOVES
        self._policy_network = SelectiveDQN(self._config.NETWORK_CONFIG)
        self._target_network = SelectiveDQN(self._config.NETWORK_CONFIG) # TODO really use the filtered network?

        self._battleState = None

        self.generate_team(teamconfig)

    def join_battle(self, battle:pp_sim.Battle, team:BattleState.Teams.TEAM1):
        self._battleState = BattleState(battle, team)

    def get_reward(self):
        if self._battleState is None:
            raise RuntimeError("Did not join a battle yet!")

        self._battleState.update()
        return self._battleState.get_reward(self._config.REWARD_FNCT)

    def select_action(self):

        if self._battleState is None:
            raise RuntimeError("Did not join a battle yet!")

        self._battleState.update()

        sample = random.random()
        eps_threshold = self._config.EPS_END + (self._config.EPS_START - self._config.EPS_END) * \
            math.exp(-1. *  self._steps_done / self._config.EPS_DECAY)
        self._steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return  self._policy_network.get_max_move(self._battleState.get_1d_tensor())
        else:
            return torch.tensor([[random.randrange(self._config.MAX_MOVES)]], device=GLOBAL_TORCH_DEVICE, dtype=torch.long)

    def optimize(self, parameter_list):
        # TODO
        pass

    def generate_team(self, config:TeamConfig = None):
        if config is None:
            config = self._teamconfig

        self._team = pp_sim.dict_to_team_set(pp_pick_six.generate_team(config.N_POKEMON))
        if not config.ALLOW_ITEMS:
            for pkm in self._team:
                pkm.item = ''

        return self._team

    