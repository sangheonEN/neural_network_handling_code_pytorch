# TSNE, PCA를 활용해서 LATENT VECTOR의 2차원으로 차원 축소한 후 LABEL을 맵핑하여 군집 분석을 수행
# BEST MODEL로 INFERENCE 결과가 얼마나 정확하게 데이터를 생성하는지 생성 이미지 시각화
import torch
from torch.utils.data import DataLoader
import model
import utils
import time
import torch.nn as nn
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np



# 학습 모델과 학습에 요구되는 lr스케줄러, 모델 파라미터를 가지는 class
class Solver(object):
    def __init__(self, args, device):

        self.args = args
        self.device = device
        self.model = model.AE(self.args)
        self.model.resume(args.save_dir, test=True)
        self.model.to(self.device)
        self.helper = utils.Helper(self.args, self.model)
        self.optimizer = self.helper.optimizer()
        self.scheduler = self.helper.scheduler()
        self.criterion = nn.MSELoss()


# Trainer의 객체는 train, valid, test data와 tensorboard summary 객체
class Evaluation(Solver):
    def __init__(self, args, device, summary, test_data):
        super(Evaluation, self).__init__(args, device)
        self.test_data = test_data
        self.summary = summary


    def tsne(self, z, y):
        tsne = TSNE(n_components=3, verbose=1, n_iter=300, perplexity=5)
        z = z.to('cpu').detach().numpy()
        y = y.to('cpu').detach().numpy()
        y = y[:300]
        tsne_v = tsne.fit_transform(z[:300])

        self.visualization(tsne_v, y)


    def visualization(self, tsne_v, y, flag_2dim = True):

        if flag_2dim == True:
            vmin = tsne_v.max()
            vmax = tsne_v.min()
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            colormapping = cm.ScalarMappable(norm=norm, cmap = cm.bwr)
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(tsne_v[:, 0], tsne_v[:, 1], c=y, cmap=cm.bwr)
            cbar = fig.colorbar(colormapping, ax=plt.gca())
            plt.show()

        else:
            vmin = tsne_v.max()
            vmax = tsne_v.min()
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            colormapping = cm.ScalarMappable(norm=norm, cmap = cm.bwr)
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(tsne_v[:, 0], tsne_v[:, 1], tsne_v[:, 2], c=y, cmap=cm.bwr)
            plt.show()


    def evaluate(self):

        start = time.time()
        print(f"Start Evaluation! time: {start}")
        print(f"-"*10)

        test_loader = DataLoader(
            self.test_data, batch_size=self.args.batch_size, shuffle=False)

        self.test(test_loader)


    def test(self, test_loader):
        self.model.eval()
        total_label = []
        total_latent_vector = []

        for step, (x, label) in enumerate(test_loader):
            x = x.view(-1, self.args.height*self.args.width).to(self.device)
            label = label.to(self.device)

            output, latent_vector = self.model(x)
            total_label.append(label)
            total_latent_vector.append(latent_vector)

        # append에 넣고 numpy로 변경하는게 올바른 방법일까? 처리 시간의 효율을 높일 수 없을까?
        total_label = torch.cat(total_label, dim=0)
        total_latent_vector = torch.cat(total_latent_vector, dim=0)

        self.tsne(total_latent_vector, total_label)

