from torch.utils.data import DataLoader
import model
import utils
import time
import torch.nn as nn
import torch
import os


# 학습 모델과 학습에 요구되는 lr스케줄러, 모델 파라미터를 가지는 class
class Solver(object):
    def __init__(self, args, device):

        self.args = args
        self.device = device
        self.model = model.AE(self.args)
        self.model.resume(args.save_dir, test=False)
        self.model.to(self.device)
        self.helper = utils.Helper(self.args, self.model)
        self.optimizer = self.helper.optimizer()
        self.scheduler = self.helper.scheduler()
        self.criterion = nn.MSELoss()


# Trainer의 객체는 train, valid data와 tensorboard summary 객체
class Trainer(Solver):
    def __init__(self, args, device, summary, train_data, valid_data):
        super(Trainer, self).__init__(args, device)
        self.train_data = train_data
        self.valid_data = valid_data
        self.summary = summary
        self.best_val_loss = float('inf')

    def Train_main(self):

        for epoch in range(self.args.max_epoch):
            self.epoch = epoch
            start = time.time()
            print(f"Start Train!, time: {start}")
            print(f"Epoch State: {epoch} / {self.args.max_epoch}")
            print(f"-"*10)

            train_loader = DataLoader(
                self.train_data, batch_size=self.args.batch_size, shuffle=True)

            valid_loader = DataLoader(
                self.valid_data, batch_size=self.args.batch_size, shuffle=False)

            self.train_epo(train_loader)
            self.valid(valid_loader)

        end = time.time()
        print(f"Finish Train!, time: {end}")

    def train_epo(self, train_loader):
        self.total_train_loss = 0.0
        self.model.train()
        for step, (x, label) in enumerate(train_loader):
            # y를 x 자기자신의 값으로 해서 본인과 똑같은 이미지를 생성하게!
            x = x.view(-1, self.args.height*self.args.width).to(self.device)
            y = x.view(-1, self.args.height*self.args.width).to(self.device)
            label = label.to(self.device)

            output, latent_vector = self.model(x)

            # loss는 MSE로 자기자신과의 PIXEL VALUE 오차를 줄이도록 학습!
            loss = self.criterion(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.total_train_loss += loss.item()

        self.total_train_loss = self.total_train_loss / len(train_loader)
        self.summary.add_scalar(
            'Train_loss_epoch', self.total_train_loss, self.epoch)

        if not os.path.exists("./save_model"):
            os.makedirs("./save_model")
        torch.save(
            {
                "epoch": self.epoch,
                "arch": self.model.__class__.__name__,
                "optim_state_dict": self.optimizer.state_dict(),
                "model_state_dict": self.model.state_dict(),
            }, os.path.join("./save_model/checkpoint.pth.tar")
        )

    def valid(self, valid_loader):

        self.model.eval()
        self.total_val_loss = 0

        for step, (x, label) in enumerate(valid_loader):
            x = x.view(-1, self.args.height*self.args.width).to(self.device)
            y = x.view(-1, self.args.height*self.args.width).to(self.device)
            label = label.to(self.device)

            output, latent_vector = self.model(x)

            loss = self.criterion(output, y)

            self.total_val_loss += loss.item()

        self.total_val_loss = self.total_val_loss / len(valid_loader)
        self.summary.add_scalar(
            "Valid_loss_epoch", self.total_val_loss, self.epoch)

        print(
            f"Validation Loss : Epoch-{self.epoch} / Val Loss-{round(self.total_val_loss, 3)}")

        if self.best_val_loss > self.total_val_loss:
            self.best_val_loss = self.total_val_loss

            if not os.path.exists("./save_model"):
                os.makedirs("./save_model")

            torch.save(
                {
                    "epoch": self.epoch,
                    "arch": self.model.__class__.__name__,
                    "optim_state_dict": self.optimizer.state_dict(),
                    "model_state_dict": self.model.state_dict(),
                }, os.path.join("./save_model/model_best.pth.tar")
            )
