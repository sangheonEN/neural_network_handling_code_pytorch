# lr_scheduler
# optimization
import torch
from torch.optim.lr_scheduler import *


class Helper:

    def __init__(self, args, model):
        self.args = args
        self.model = model

    def optimizer(self):

        if self.args.optim == 'adam':

            self.optim = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.lr,
                                          weight_decay=self.args.weight_decay)

        else:
            self.optim = torch.optim.SGD(self.model.parameters(),
                                         lr=self.args.lr,
                                         weight_decay=self.args.weight_decay,
                                         momentum=self.args.momentum)

        return self.optim

    def scheduler(self):

        # total train data num 42,000. batch 32. iter/epoch = 1312
        scheduler = StepLR(self.optim, step_size=10000, gamma=0.5)

        return scheduler
