from src.utils.get_torch_device import GLOBAL_TORCH_DEVICE
from src.utils.states import PokemonState
import torch

def reward_min_enemy_hp(own_state:PokemonState, opponent_state:PokemonState) -> torch.tensor:
    result = (torch.tensor(1.0, device=GLOBAL_TORCH_DEVICE) -\
              torch.tensor(opponent_state.hprel, device=GLOBAL_TORCH_DEVICE))
    return result.reshape(1)

def reward_hp_diff( own_state:PokemonState, opponent_state:PokemonState,\
                    alpha:torch.tensor=torch.tensor(1.0, device=GLOBAL_TORCH_DEVICE),\
                    beta:torch.tensor=torch.tensor(1.0, device=GLOBAL_TORCH_DEVICE)\
                    ) -> torch.tensor:

    result =  (alpha*torch.tensor(own_state.hprel, device=GLOBAL_TORCH_DEVICE) -\
               beta*torch.tensor(opponent_state.hprel, device=GLOBAL_TORCH_DEVICE))
    return result.reshape(1)

def reward_step_two_enemy_hp(own_state:PokemonState, opponent_state:PokemonState) -> torch.tensor:
    r = 0.0
    if opponent_state.hprel <= 0.0:
        r = 1.0
    elif opponent_state.hprel < 0.5:
        r = 0.5
    return torch.tensor([r], device=GLOBAL_TORCH_DEVICE)


def reward_step_one_enemy_hp(own_state:PokemonState, opponent_state:PokemonState) -> torch.tensor:
    r = 0.0
    if opponent_state.hprel <= 0.0:
        r = 1.0
    
    return torch.tensor([r], device=GLOBAL_TORCH_DEVICE)