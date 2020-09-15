from src.utils.get_torch_device import GLOBAL_TORCH_DEVICE
from src.utils.states import PokemonState

def reward_hp_diff(own_state:PokemonState, opponent_state:PokemonState, alpha=1.0, beta=1.0):
    # TODO use tensors?
    return (alpha*own_state.hprel - beta*opponent_state.hprel)