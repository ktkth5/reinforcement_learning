

class AverageMeter:

    def __init__(self):
        self.reset()

    def update(self, value, n=1):
        self.count += n
        self.value = value
        self.sum += value * n
        self.avg = self.sum / self.count

    def reset(self):
        self.count = 0
        self.sum = 0
        self.value = 0
        self.avg = 0