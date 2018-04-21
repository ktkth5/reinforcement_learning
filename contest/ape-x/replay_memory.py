import random
import torch.nn as nn


# clear
class ReplayMemory(nn.Module):

    def __init__(self, capacity):
        super(ReplayMemory, self).__init__()
        self.capacity = capacity
        self.memory = []

    def push(self, priority, error):
        if len(self.memory) < self.capacity:
            self.memory.append({"priority":priority, "error":error})
        else:
            pass

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


if __name__=="__main__":
    rm = ReplayMemory(5)
    for i in range(10):
        x = random.randint(0,5)
        y = random.randint(5,10)
        rm.push(x,y)
        if i % 3 == 0 and len(rm)>2:
            print(rm.sample(2))