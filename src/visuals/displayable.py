
from pokemonpython.sim.structs import Battle
class IDisplayable(object):
    def __init__(self):
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError

    def set_battle(self, battle:Battle):
        raise NotImplementedError

    def start_turn(self, team_1_mv_idx:int, team_2_mv_idx:int):
        raise NotImplementedError

    def end_turn(self):
        raise NotImplementedError

class DisplayableStub(object):
    def __init__(self):
        pass

    def initialize(self):
        pass

    def set_battle(self, battle:Battle):
        pass

    def start_turn(self, team_1_mv_idx:int, team_2_mv_idx:int):
        pass

    def end_turn(self):
        pass