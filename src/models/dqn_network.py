from dataclasses import dataclass, field
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.get_torch_device import GLOBAL_TORCH_DEVICE
from src.utils.states import PokemonState

@dataclass
class SelectiveDQNConfig(object):
    D_IN:int # Input length of network
    D_OUT:int # out put length of network
    HIDDENLAYER_SIZES = [32,32,32] #TODO sizes
    HIDDENLAYER_NETWORK_TYPE = torch.nn.Linear # layer type
    ACTIVATION_FUNCTION = F.relu # activator function of each layer
    AVAILABLE_MOVE_IDS:List[int] = field(default_factory=list)
    
class SelectiveDQN(nn.Module):
    def __init__(self, config:DQNConfig):
        super(DQN, self).__init__()
        self._config = config
        self._network = DQN(config.D_IN, config, config.D_OUT)
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
    def __init__(self, D_in:int, hiddenlayer_sizes:List[int], D_out:int, hiddenlayer_type = torch.nn.Linear, actFcnt = F.Relu):
        super(DQN, self).__init__()

        self._layers = []
        self._actFcnt = actFcnt


        self._layers = [D_in].append(hiddenlayer_sizes).append(D_out)
        for l_idx in range(len(self._layers)-1):
            self.self._layers.append( hiddenlayer_type(self._layers[l_idx],self._layers[l_idx+1]) )

        self._head = self._layers[-1]
        
    def forward(self, x):
        for l in self.self._layers[:-1]:
            x = self._actFcnt ( l(x) )
        # no activation function for output layer
        return self._head(x)
        

