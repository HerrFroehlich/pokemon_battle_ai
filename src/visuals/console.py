from src.visuals.displayable import IDisplayable
from src.utils.statistics import calc_modifier
from pokemonpython.sim.structs import Battle
import pokemonpython.data.dex as dex
import time
import math

class ConsoleDisplayable(IDisplayable):
    def __init__(self, sleep_on_end_ms:int = 0):
        self._sleep_on_end_ms = sleep_on_end_ms
        self._battle = None

    def initialize(self):
        pass

    def set_battle(self, battle:Battle):
        self._battle = battle

    def start_turn(self, team_1_mv_idx:int, team_2_mv_idx:int):
        if self._battle == None:
            raise RuntimeError("No battle was set yet!")

        print ("*xxxxxxxxxxxxx TURN %d xxxxxxxxxxxxx* " % (self._battle.turn+1))
            
        pkm1 = self._battle.p1.active_pokemon[0]
        pkm2 = self._battle.p2.active_pokemon[0]
        for i in range(1,3):

            if i == 1:
                pkm =  pkm1
                enemy_pkm = pkm2
                team_mv = team_1_mv_idx
            else:
                pkm =  pkm2
                enemy_pkm = pkm1
                team_mv = team_2_mv_idx

            if team_mv == None:
                print("#-- TEAM %d passed" % i)
            else:
                mvname = pkm.moves[team_mv]
                mvtype = dex.move_dex[mvname].type
                print("#------- %s -------#" % pkm.name)
                print("Hp %d/%d" % (pkm.hp, pkm.maxhp))
                print("pp %s" % (pkm.pp))
                print("Types %s" % pkm.types)
                print("Status %s" % pkm.status)
                print("Using move %s which has type %s" % (mvname, mvtype))

                mult = calc_modifier(mvname, pkm, enemy_pkm)
                if math.isnan(mult):
                    print("Used move is a status move")
                else:
                    print("Used moved has multiplier %d against enemy type %s" % (mult, enemy_pkm.types) )

    def end_turn(self):
        
        if self._battle.ended:
            pkm1 = self._battle.p1.active_pokemon[0]
            pkm2 = self._battle.p2.active_pokemon[0]
            print("#------- %s -------#" % pkm1.name)
            print("Hp %d/%d" % (pkm1.hp, pkm1.maxhp))
            print("#------- %s -------#" % pkm2.name)
            print("Hp %d/%d" % (pkm2.hp, pkm2.maxhp))
            if self._battle.winner == 'p1':
                print ("*xxxxxxxxxxxxx BATTLE WON BY TEAM 1 xxxxxxxxxxxxx* ")
            elif self._battle.winner == 'p2':
                print ("*xxxxxxxxxxxxx BATTLE WON BY TEAM 2 xxxxxxxxxxxxx* ")
            else:
                print ("*xxxxxxxxxxxxx BATTLE WON BY TEAM 2 xxxxxxxxxxxxx* ")

        if self._sleep_on_end_ms > 0:
            time.sleep(self._sleep_on_end_ms/1000)