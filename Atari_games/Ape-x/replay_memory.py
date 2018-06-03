import random
import torch
import torch.nn as nn
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

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
            self.memory.pop(0)
            self.memory.append({"priority":priority, "error":error})
            pass

    def priority_sample(self, batch_size):
        assert len(self.memory)>=batch_size

        for i, data in enumerate(self.memory):
            # print(data["priority"])
            if data["priority"].item()<0:
                data["priority"] = torch.tensor(0.01, dtype=torch.float)
            if i==0:
                priority = data["priority"].unsqueeze(0)
            else:
                # print("priority", priority.shape)
                # print("data[p]", data["priority"].unsqueeze(0).shape)
                priority = torch.cat((priority, data["priority"].unsqueeze(0)))
        sample_index = list(map(int,torch.multinomial(priority, batch_size)))
        # print(sample_index)
        test_list = []
        try:
            for i, index in enumerate(sorted(sample_index)):
                index =+ i
                if i==0:
                    # sample_error_batch = self.memory[index]["error"].unsqueeze(0)
                    error = self.memory[index]["error"]
                    # self.memory[index]["priority"] = torch.tensor(0., dtype=torch.float)
                    self.memory.pop(index)
                else:
                    # sample_error_batch = torch.cat((sample_error_batch,
                    #                                 self.memory[index]["error"].unsqueeze(0)))
                    # self.memory[index]["priority"] = torch.tensor(0., dtype=torch.float)
                    error += self.memory[index]["error"]
                    self.memory.pop(index)
        except:
            for i, index in enumerate(sample_index):
                test_list.append(float(self.memory[index]["priority"]))
            print(test_list)

        # print(sample_error_batch)
        # return sample_error_batch
        # print(error.requires_grad)
        return error

    def reset(self):
        self.memory = []

    def __len__(self):
        return len(self.memory)


if __name__=="__main__":
    rm = ReplayMemory(6)
    for i in range(10):
        x = torch.tensor([random.randint(1,5)], dtype=torch.float)
        y = torch.FloatTensor([random.randint(5,10)])
        rm.push(x,y)

    print(rm.priority_sample(2))
