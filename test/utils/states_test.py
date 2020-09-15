
import pokemonpython.sim.sim as sim
import numpy as np
from pokemonpython.tools.pick_six import generate_team
from src.utils.states import BattleState, PokemonState
from pokemonpython.sim.sim import new_battle, do_turn
from pokemonpython.sim.player import default_decide

team = sim.dict_to_team_set(generate_team(1))

battle = new_battle('single', "Team1", team, "Team2", team, debug=True) # single 1vs1 , double: 2vs2 (same time)

pkm = battle.p1.active_pokemon[0]

s = PokemonState(pkm)

print(s)
print(s.available_moves)

pkm.hp = 3
pkm.status = 'frz'
pkm.pp[s.available_moves[0]] = 0

s = PokemonState(pkm)

print(s)
print(s.available_moves)

print(PokemonState.get_tensor_size())
print(PokemonState.get_tensor_regions())

bs = BattleState(battle)

print (bs)
default_decide(battle.p1)
default_decide(battle.p2)
do_turn(battle)
bs.update()
print(bs)
print(bs.to_1d_tensor())