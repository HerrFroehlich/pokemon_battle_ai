from dataclasses import dataclass, field
import torch
import random
import math

import pokemonpython.sim.sim as pp_sim
import pokemonpython.tools.pick_six as pp_pick_six

from src.utils.get_torch_device import GLOBAL_TORCH_DEVICE
from src.utils.replay_memory import ReplayMemory, Transition
from src.utils.states import BattleState, PokemonState
from src.models.dqn_network import DQNConfig, DQN
from src.models.rewards import reward_hp_diff

@dataclass
class AgentConfig(object):
    MEMORY_SIZE = 10000 # NOF stored states in memory
    BATCH_SIZE = 128 # NOF batches used for optimizing
    EPS_START = 0.9 # epsilon start for expontential random decay function
    EPS_END = 0.05# epsilon end for expontential random decay function
    EPS_DECAY = 200 # decay gradient for expontential random decay function
    GAMMA = 0.999
    TARGET_UPDATE = 10
    MAX_MOVES = 4
    NETWORK_CONFIG = DQNConfig()
    REWARD_FNCT:object = field(default=reward_hp_diff)
    LOSS_FNCT:object = field(default=torch.nn.functional.smooth_l1_loss)
    OPTIMIZER = torch.optim.RMSprop

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
        self._policy_network = DQN(self._config.NETWORK_CONFIG)
        self._target_network = DQN(self._config.NETWORK_CONFIG) # TODO really use the filtered network?
        self._target_network.load_state_dict(self._policy_network.state_dict())
        self._target_network.eval()

        self._optimizer = self._config.OPTIMIZER(self._policy_network.parameters())

        self._battleState = None
        self._memory = ReplayMemory(self._config.MEMORY_SIZE)

        self._lastaction = None

        self.generate_team(teamconfig)

    def join_battle(self, battle:pp_sim.Battle, team:BattleState.Teams):
        self._battleState = BattleState(battle, team)

    def end_turn(self) -> torch.FloatTensor:
        """Ends a turn, calculates the reward and stores the previous
         and current state, the last action and the reward to the replay memory

        Raises:
            RuntimeError: If join_battle was not called yet
            RuntimeError: if select_action was not called yet

        Returns:
            float: reward for last action
        """
        if self._battleState is None:
            raise RuntimeError("Did not join a battle yet!")

        if self._lastaction is None:
            raise RuntimeError("Did not  perform an action yet!")

        old_state = self._battleState.to_1d_tensor()
        self._battleState.update()
        reward = self._battleState.get_reward(self._config.REWARD_FNCT)
        new_state = self._battleState.to_1d_tensor()

        self._memory.push(old_state, self._lastaction, new_state, reward)

        return reward


    def select_action(self) -> (str, int):

        if self._battleState is None:
            raise RuntimeError("Did not join a battle yet!")

        self._battleState.update()

        sample = random.random()
        eps_threshold = self._config.EPS_END + (self._config.EPS_START - self._config.EPS_END) * \
            math.exp(-1. *  self._steps_done / self._config.EPS_DECAY)
        self._steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                self._lastaction = self._get_max_move(self._battleState.to_1d_tensor())
                #TODO what if all moves nodes have <0?
        else:
            self._lastaction = torch.tensor([[random.randrange(self._config.MAX_MOVES)]], device=GLOBAL_TORCH_DEVICE, dtype=torch.long)

        # in this context this is the new action
        # TODO implement pass
        return "move", self._lastaction.item()
    

    def optimize(self):
        if len(self._memory) < self._config.BATCH_SIZE:
            print("Not enough batches yet")
            return

        transitions = self._memory.sample(self._config.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # # Compute a mask of non-final states and concatenate the batch elements
        # # (a final state would've been the one after which simulation ended)
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                     batch.next_state)), device=device, dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch.next_state
        #                                             if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_next_batch = torch.cat(batch.next_state)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self._policy_network(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        # next_state_values = torch.zeros(self._config.BATCH_SIZE, device=device)
        next_state_values = self._target_network(state_next_batch).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self._config.GAMMA) + reward_batch
        # Compute Huber loss
        loss = self._config.LOSS_FNCT(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        # for param in policy_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self._optimizer.step()

    def generate_team(self, config:TeamConfig = None):
        if config is None:
            config = self._teamconfig

        self._team = pp_sim.dict_to_team_set(pp_pick_six.generate_team(config.N_POKEMON))
        if not config.ALLOW_ITEMS:
            for pkm in self._team:
                pkm.item = ''

        return self._team


    def _get_max_move(self, state):
        # only evaluate valid moves
        filtered_output = self._battleState.filter_move_tensor_by_available(self._policy_network(state))
        # find the best move of all batches (dim=1)
        # TODO what if there is none?
        # 1) no more moves
        # 2)or all yield 0,0,0,0 ? -> most likely shouldn't happen thanks to random moves

        # t.max(1) will return largest column value of each row. (so over batches)
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        return filtered_output.max(1)[1].view(1, 1)