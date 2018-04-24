import random
import torch
import torch.nn as nn
from torch.autograd import Variable


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

    def priority_sample(self, batch_size):

        for i, data in enumerate(self.memory):
            if i==0:
                priority = data["priority"]
            else:
                priority = torch.cat((priority, data["priority"]), 0)

        sample_index = list(map(int,torch.multinomial(priority, batch_size)))

        for i, index in enumerate(sample_index):
            if i==0:
                sample_error_batch = self.memory[index]["error"]
                error = self.memory[index]["error"]
                self.memory[index]["priority"] = Variable(torch.FloatTensor([0.01]))
            else:
                sample_error_batch = torch.cat((sample_error_batch, self.memory[index]["error"]))
                self.memory[index]["priority"] = Variable(torch.FloatTensor([0.01]))
                error += self.memory[index]["error"]

        # print(sample_error_batch)
        # return sample_error_batch
        return error

    def __len__(self):
        return len(self.memory)


if __name__=="__main__":
    rm = ReplayMemory(6)
    for i in range(10):
        x = torch.FloatTensor([random.randint(1,5)])
        y = torch.FloatTensor([random.randint(5,10)])
        rm.push(x,y)
    print(rm.priority_sample(2))
