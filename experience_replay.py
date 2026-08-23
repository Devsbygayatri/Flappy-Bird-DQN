from collections import deque
import random 

class ReplayMemory():

    #create FIFO queue - experience replay
    def __init__(self, maxlen, seed = None):
        self.memory = deque([], maxlen = maxlen)

    def append(self, new_exp): #add elements 
        self.memory.append(new_exp)

    def sample(self, sample_size): #extract random samples
        return random.sample(self.memory, sample_size)

    def __len__(self): #cal length of memory / curr buffer size
        return len(self.memory)