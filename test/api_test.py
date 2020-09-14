import sys; print(sys.path)
import pokemonpython.sim.sim as sim
import numpy as np
from pokemonpython.tools.pick_six import generate_team
import pokemonpython.data.dex as dex
import os
import json
from pokemonpython.sim.sim import new_battle, run


with open(os.path.join(os.path.dirname(__file__), '../pokemonpython/data/moves.json')) as f:
    moves_raw_data = json.load(f)

print("------ 1vs1 Battle ------")

teams = []
for i in range(2):
    teams.append(sim.dict_to_team_set(generate_team(1)))

print ("TEAM 0 Pokemons")
# print (teams[0])
print ("Name: %s, Moves: %s" % (teams[0][0].name, teams[0][0].moves))

print ("TEAM 1 Pokemons")
# print (teams[1])
print ("Name: %s, Moves: %s" % (teams[1][0].name, teams[1][0].moves))

print("Default moves")
battle = new_battle('single', "Team1", teams[0], "Team2", teams[1], debug=True) # single 1vs1 , double: 2vs2 (same time)
run(battle)

# random choices
print("Random choices")

battle = new_battle('single', "Team1", teams[0], "Team2", teams[1], debug=True) # single 1vs1 , double: 2vs2 (same time)
while not battle.ended:
    pkm1 = battle.p1.active_pokemon[0]
    pkm2 = battle.p2.active_pokemon[0]
    for i in range(1,3):
        mv = np.random.randint(0,3)
        choicestr = 'move ' + str(mv)
        sim.choose(battle,i, choicestr)

        if i == 1:
            pkm =  pkm1
            enemy_pkm = pkm2
        else:
            pkm =  pkm2
            enemy_pkm = pkm1

        print("#------- %s -------#" % pkm.name)
        print("Hp %d/%d" % (pkm.hp, pkm.maxhp))
        print("pp %s" % (pkm.pp))
        print("Types %s" % pkm.types)
        print("Status %s" % pkm.status)
        mvname = pkm.moves[mv]
        mvtype = dex.move_dex[mvname].type
        print("Using move %s which has type %s" % (mvname, mvtype))

        mult = 0
        typestr = ''
        for t in enemy_pkm.types:
            m = dex.typechart_dex[t].damage_taken[mvtype]
            if m > mult:
                typestr = t
                mult = m

        print("Used moved has multiplier %d against enemy type %s" % (mult, typestr) )

    sim.do_turn(battle)
        # c = dex.Decision('move', mv)
        # if i == 0:
        #         battle.p1.choice = c
        # if i == 1:
        #         battle.p2.choice = c