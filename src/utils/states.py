from collections import namedtuple
from pokemonpython.sim.structs import Pokemon, Battle
from pokemonpython.data.dex import pokedex
from enum import Enum, auto
from typing import List
from torch import Tensor

class PokemonState(object):
        
    class Status(Enum):
        NORMAL = auto()
        POISONED = auto()
        BADLY_POISONED = auto()
        SLEEPING = auto()
        FROZEN = auto()
        BURNED = auto()
        PARALYZED = auto()
        @staticmethod
        def fromStr(stateStr:str):
            if (stateStr == 'brn'):
                return PokemonState.Status.BURNED
            elif (stateStr == 'par'):
                return PokemonState.Status.PARALYZED
            elif (stateStr == 'psn'):
                return PokemonState.Status.POISONED
            elif (stateStr == 'slp'):
                return PokemonState.Status.SLEEPING
            elif (stateStr == 'tox'):
                return PokemonState.Status.BADLY_POISONED
            elif (stateStr == 'frz'):
                return PokemonState.Status.FROZEN
            else:
                return PokemonState.Status.NORMAL
    
    MAX_POKEDEX_INDEX = len(pokedex)
    MAX_MOVES = 4
    MAX_STATES = len(Status)

    hprel:float = 0.0
    status:Status = Status.NORMAL
    available_moves:List[int] = []
    available_move_ids:List[int] = []
    pokedex_id:int = 0

    def __init__(self, pkm:Pokemon):
        self.hprel = pkm.hp / pkm.maxhp
        self.status = PokemonState.Status.fromStr(pkm.status)

        self.available_moves = []
        for (key, val) in pkm.pp.items():
            if (val != 0):
                self.available_moves.append(key)
        self.available_move_ids = []
        for mv in self.available_moves:
            self.available_move_ids.append(pkm.moves.index(mv))
        # Map a unique id to each pokemon (to generate input neurons later)
        self.pokedex_id = list(pokedex.keys()).index(pkm.id)

    def __str__(self):
        str =  '''\
#------------
# ID: %d
# relative HP: %.2f
# Status: %s
# Available Moves: %s
# Available Move IDs: %s
#------------ \n''' % (self.pokedex_id, self.hprel, self.status, self.available_moves, self.available_move_ids)
        return str

    @staticmethod
    def get_tensor_size() -> int:
        return PokemonState.MAX_POKEDEX_INDEX + PokemonState.MAX_MOVES + PokemonState.MAX_STATES + 1 # relative HP
               

    def to_tensor(self) -> Tensor:
        pass



class BattleState(object):

    class Variant(Enum):
        SIMPLE = auto() # use only pkm id, status, relative hp

    _variant:Variant = Variant.SIMPLE
    _battle:Battle = None
    _team1_active_pkm_state:PokemonState = None
    _team2_active_pkm_state:PokemonState = None

    def update(self):
        self._team1_active_pkm_state = PokemonState(self._battle.p1.active_pokemon[0]) # TODO Double fights
        self._team2_active_pkm_state = PokemonState(self._battle.p2.active_pokemon[0]) # TODO Double fights
    
    def get_tensor_size(self) -> int:
        if (self._variant == BattleState.Variant.SIMPLE):
            return 2*PokemonState.get_tensor_size()
        else:
            return 0

    def to_tensor(self) -> Tensor:
        pass

    def __init__(self, battle:Battle, variant:Variant = Variant.SIMPLE):
        self._battle = battle
        self._variant = variant
        self.update()

    
    def __str__(self):
        return " *Active Pokemon of Team 1:*\n" + \
            str(self._team1_active_pkm_state) + \
            " *Active Pokemon of Team 2:*\n" + \
            str(self._team2_active_pkm_state)
