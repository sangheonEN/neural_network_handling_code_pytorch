from torch.utils.data import DataLoader
import model
import utils
import time
# 학습 모델과 학습에 요구되는 lr스케줄러, 모델 파라미터를 가지는 class


class Solver(object):
    def __init__(self, args, device):

        self.args = args
        self.device = device
        self.model = model.AE
        self.model.to(self.device)
        self.helper = utils.Helper()
        self.optimizer = self.helper.optimizer()
        self.scheduler = self.helper.scheduler()


class Trainer(Solver):
    def __init__(self, args, device, x_train, y_train, x_valid, y_valid):
        super(Trainer, self).__init__(args, device)
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

    def Train_epo(self):

        for epoch in range(self.args.max_epoch):
            start = time.time()
            print(f"Epoch State: {epoch} / {self.args.max_epoch}")
            print(f"-"*10)

            train_x, train_y = DataLoader(self.x_train, self.y_train,
                                          batch_size=self.args.batch_size, shuffle=True)

    def valid(self):
        valid_x, valid_y = DataLoader(self.x_valid, self.y_valid,
                                      batch_size=self.args.batch_size, shuffle=False)


if __name__ == "__main__":
    pass
