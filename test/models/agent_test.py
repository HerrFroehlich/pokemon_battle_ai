
from src.models.agent import Agent, AgentConfig
from pokemonpython.sim.sim import new_battle, run
import pokemonpython.sim.sim as sim
import numpy as np
from pokemonpython.tools.pick_six import generate_team
from src.utils.states import BattleState, PokemonState
from pokemonpython.sim.sim import new_battle, do_turn
from pokemonpython.sim.player import default_decide
import pokemonpython.data.dex as dex

conf = AgentConfig()
conf.BATCH_SIZE = 3

a1 = Agent(conf)
t1 = a1.generate_team()
a2 = Agent(conf)
t2 = a2.generate_team()

agents = [a1, a2]

print(t1)

battle = new_battle('single', "Team1", t1, "Team2", t2, debug=True) # single 1vs1 , double: 2vs2 (same time)
a1.join_battle(battle, 1)
a2.join_battle(battle, 2)

while not battle.ended:
    pkm1 = battle.p1.active_pokemon[0]
    pkm2 = battle.p2.active_pokemon[0]
    for i in range(1,3):
        actionstr, mv = agents[i-1].select_action()
        choicestr = actionstr + ' ' + str(mv)
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
        print("Team %d : reward %0.2f" % (i, agents[i-1].end_turn()))
        agents[i-1].optimize()

        print("Used moved has multiplier %d against enemy type %s" % (mult, typestr) )

    sim.do_turn(battle)