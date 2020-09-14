import numpy as np
from numpy_ringbuffer import RingBuffer
import random

# TODO transition class
from collections import namedtuple
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = RingBuffer(capacity, dtype=object)

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))
    

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)