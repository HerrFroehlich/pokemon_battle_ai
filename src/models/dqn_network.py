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