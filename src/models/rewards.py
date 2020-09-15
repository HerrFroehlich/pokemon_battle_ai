from src.utils.get_torch_device import GLOBAL_TORCH_DEVICE
from src.utils.states import PokemonState
import torch

def reward_hp_diff(own_state:PokemonState, opponent_state:PokemonState,\
        alpha:torch.tensor=torch.tensor(1.0, device=GLOBAL_TORCH_DEVICE),\
        beta:torch.tensor=torch.tensor(1.0, device=GLOBAL_TORCH_DEVICE)\
        ) -> torch.tensor:
    result =  (alpha*torch.tensor(own_state.hprel, device=GLOBAL_TORCH_DEVICE) -\
        beta*torch.tensor(opponent_state.hprel, device=GLOBAL_TORCH_DEVICE))

    return result.reshape(1)