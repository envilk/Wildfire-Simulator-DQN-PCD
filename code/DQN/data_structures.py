# python libraries

import collections

# own python modules

from common_fixed_variables import *


class ReplayMemory(object):

    def __init__(self, capacity, transition):
        self.memory = collections.deque([], maxlen=capacity)
        self.transition = transition

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition(*args))

    def sample(self):
        return SYSTEM_RANDOM.sample(self.memory, BATCH_SIZE)

    def __len__(self):
        return len(self.memory)