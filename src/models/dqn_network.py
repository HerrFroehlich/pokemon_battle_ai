from dataclasses import dataclass, field
from typing import List
import torch
import torch.nn as nn

from src.utils.get_torch_device import GLOBAL_TORCH_DEVICE
from src.utils.states import PokemonState

@dataclass
class DQNConfig(object):
    D_IN:int = 1024# Input length of network
    D_OUT:int = 1024# out put length of network
    HIDDENLAYER_SIZES = [512,256,32,256,512] #TODO sizes
    HIDDENLAYER_NETWORK_TYPE = torch.nn.Linear # layer type
    ACTIVATION_FUNCTION:object = field(default=torch.nn.functional.relu) # activator function of each layer
    
class SelectiveDQN(nn.Module):
    def __init__(self, config:DQNConfig):
        super(DQN, self).__init__()
        self._config = config
        self._network = DQN(config.D_IN, config, config.D_OUT).to(GLOBAL_TORCH_DEVICE)
        self._move_ids = torch.LongTensor(config.AVAILABLE_MOVE_IDS)-1 # move ids start with 1
        # n_moves = len(config.AVAILABLE_MOVE_IDS)
        # values = torch.ones(n_moves, device=GLOBAL_TORCH_DEVICE)
        # self._move_mask = torch.sparse_coo_tensor(config.AVAILABLE_MOVE_IDS, values, \
        #     torch.Size(n_moves), device=GLOBAL_TORCH_DEVICE)

    def set_available_move_ids(ids:List[int]):
        self._move_ids = torch.LongTensor(ids)-1 # move ids start with 1


    def get_max_move(self, state):
        # only evaluate valid moves
        filtered_output = self._network(state).index_select(0, self._move_ids)
        # find the best move of all batches (dim=1)
        # TODO what if there is none? aka all yield 0,0,0,0 ? -> most likely shouldn't happen thanks to random moves

        # t.max(1) will return largest column value of each row. (so over batches)
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        return filtered_output.max(1)[1].view(1, 1)



class DQN(nn.Module):
    def __init__(self,config:DQNConfig):
        super(DQN, self).__init__()
        self._config = config
        # n_moves = len(config.AVAILABLE_MOVE_IDS)
        # values = torch.ones(n_moves, device=GLOBAL_TORCH_DEVICE)
        # self._move_mask = torch.sparse_coo_tensor(config.AVAILABLE_MOVE_IDS, values, \
        #     torch.Size(n_moves), device=GLOBAL_TORCH_DEVICE)

        self._layers = []
        self._actFcnt = config.ACTIVATION_FUNCTION

        self._layer_sizes = [config.D_IN]
        self._layer_sizes += config.HIDDENLAYER_SIZES
        self._layer_sizes.append(config.D_OUT)
        for l_idx in range(len(self._layer_sizes)-1):
            self._layers.append( config.HIDDENLAYER_NETWORK_TYPE(self._layer_sizes[l_idx],self._layer_sizes[l_idx+1]).to(GLOBAL_TORCH_DEVICE) )

        self._head = self._layers[-1]
        
    def forward(self, x):
        for l in self._layers[:-1]:
            x = self._actFcnt ( input = l(x) )
        # no activation function for output layer
        return self._head(x)