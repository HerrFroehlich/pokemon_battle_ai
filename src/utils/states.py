from collections import namedtuple
from pokemonpython.sim.structs import Pokemon, Battle
from pokemonpython.data.dex import pokedex, move_dex
from enum import Enum, IntEnum, auto
from typing import List, DefaultDict
from torch import Tensor
import torch

class PokemonState(object):
        
    class Status(IntEnum):
        NORMAL = 0
        POISONED = 1
        BADLY_POISONED = 2
        SLEEPING = 3
        FROZEN = 4
        BURNED = 5
        PARALYZED = 6
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
    MAX_MOVES = len(move_dex)
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
            self.available_move_ids.append(move_dex[mv].num)
            # self.available_move_ids.append(id_to_index_moves[mv])
        # Map a unique id to each pokemon (to generate input neurons later)
        self.pokedex_id = pokedex[pkm.id].num       # TODO pokemon with negative nums?
        if ((self.pokedex_id) < 0):
            raise NotImplementedError("What shall do we do with negative numbers? Early in the morning")


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
    
    @staticmethod
    def get_tensor_regions() -> DefaultDict["string", List[int]]:
        d = {}
        d["Pokemons"] =  [0, PokemonState.MAX_POKEDEX_INDEX-1]
        offset = PokemonState.MAX_POKEDEX_INDEX
        d["Moves"] =  [offset, offset+PokemonState.MAX_MOVES-1]
        offset += PokemonState.MAX_MOVES
        d["States"] =  [offset, offset+PokemonState.MAX_STATES-1]
        offset += PokemonState.MAX_STATES
        d["relative HP"] =  [offset]
        #offset += 1
        return d

    def to_1d_tensor(self, offset:int = 0) -> torch.sparse_coo_tensor:
        # sparse_tensor = zeros(PokemonState.get_tensor_size())
        # Activate the respective pokemons neuron
        # if ((offset + PokemonState.get_tensor_size()) > len(sparse_tensor)):
        #     raise IndexError("Writting to that tensor beginning from offset %d exceeds its size %d" % (offset, len(sparse_tensor)))

        idx = [[offset + self.pokedex_id-1]] # pkm ids start with idx 1
        values = [1]
        offset = PokemonState.MAX_POKEDEX_INDEX
        for mv_id in self.available_move_ids:
            idx.append([offset + mv_id - 1]) # mv ids start with idx 1
            values.append(1)

        offset += PokemonState.MAX_MOVES
        idx.append([offset+self.status])
        values.append(1)
    
        offset += PokemonState.MAX_STATES
        idx.append([offset])
        values.append(self.hprel)

        idx = torch.LongTensor(idx).t()
        # idx.resize(1, len(values))
        values = torch.FloatTensor(values)
        sparse_tensor = torch.sparse_coo_tensor(idx, values, torch.Size([PokemonState.get_tensor_size()]))
        # sparse_tensor[idx] = values

        return sparse_tensor




class BattleState(object):

    class Variant(Enum):
        SIMPLE = auto() # use only pkm id, status, relative hp

    s_variant:Variant = Variant.SIMPLE
    _battle:Battle = None
    _team1_active_pkm_state:PokemonState = None
    _team2_active_pkm_state:PokemonState = None

    def update(self):
        self._team1_active_pkm_state = PokemonState(self._battle.p1.active_pokemon[0]) # TODO Double fights
        self._team2_active_pkm_state = PokemonState(self._battle.p2.active_pokemon[0]) # TODO Double fights
    
    def get_tensor_size(self) -> int:
        if (BattleState.s_variant == BattleState.Variant.SIMPLE):
            return 2*PokemonState.get_tensor_size()
        else:
            return 0

    def to_1d_tensor(self) ->  torch.sparse.FloatTensor:
        tensor = torch.sparse.FloatTensor(self.get_tensor_size())
        
        #Add pokemon states
        tensor = self._team1_active_pkm_state.to_1d_tensor()
        # offset = self._team1_active_pkm_state.get_tensor_size()
        tensor = torch.cat([tensor, self._team2_active_pkm_state.to_1d_tensor()])
        return tensor


    def __init__(self, battle:Battle):
        self._battle = battle
        self.update()

    
    def __str__(self):
        return " *Active Pokemon of Team 1:*\n" + \
            str(self._team1_active_pkm_state) + \
            " *Active Pokemon of Team 2:*\n" + \
            str(self._team2_active_pkm_state)
