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
import cv2
import os
from PIL import Image


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
    def __init__(self, args, device, summary, test_data, valid_data):
        super(Evaluation, self).__init__(args, device)
        self.test_data = test_data
        self.summary = summary
        self.valid_data = valid_data


    def input_output_visualization(self, input_arr, output_arr):
        input = input_arr.to('cpu').detach().numpy()
        output = output_arr.to('cpu').detach().numpy()

        # input data 0~1 사이의 값으로 다시 복원 (transform 가우시안 정규화를 다시 복원함)
        # mean = 0.1307
        # std = 0.3081
        # input = input * std + mean

        # plot parameters
        cols, rows = 15, 20
        fig, axs = plt.subplots(rows, cols)
        inputs = []

        for r in range(rows):
            for c in range(cols):
                axs[r, c].set(yticklabels=[])
                axs[r, c].set(xticklabels=[])
                axs[r, c].tick_params(left=False)
                axs[r, c].tick_params(right=False)
                inputs.append(axs[r, c].imshow(input[r+c].reshape(28, 28, 1), cmap='gray'))
                # axs[r, c].label_outer()


        plt.show()

        fig2, axs2 = plt.subplots(rows, cols)
        outputs = []

        for r in range(rows):
            for c in range(cols):
                axs2[r, c].set(yticklabels=[])
                axs2[r, c].set(xticklabels=[])
                axs2[r, c].tick_params(left=False)
                axs2[r, c].tick_params(right=False)
                outputs.append(axs2[r, c].imshow(output[r+c].reshape(28, 28, 1), cmap='gray'))
                # axs2[r, c].label_outer()

        plt.show()


    def tsne(self, z, y, input, output):
        # perplexity: manifold learning의 nearest neighbors 갯수에 사용되는 값을 뜻합니다. 일반적으로 더 큰 데이터 셋은 보통 더 큰 perplexity 값을 필요. (5~50 사이의 값으로 실험)
        # verbose: [1: True, 0: False]TSNE가 진행 결과 나오게 할 것인지 결정하는 파라미터.
        tsne = TSNE(n_components=3, verbose=2, n_iter=300, perplexity=20)
        z = z.to('cpu').detach().numpy()
        y = y.to('cpu').detach().numpy()
        y = y[:6000]
        tsne_v = tsne.fit_transform(z[:6000])

        self.tsne_visualization(tsne_v, y)
        self.input_output_visualization(input, output)


    def tsne_visualization(self, tsne_v, y, flag_2dim = False):

        if flag_2dim == True:
            # vmin = tsne_v.max()
            # vmax = tsne_v.min()
            # norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            # colormapping = cm.ScalarMappable(norm=norm, cmap = cm.bwr)
            fig = plt.figure(figsize=(6, 6))
            plt.scatter(tsne_v[:, 0], tsne_v[:, 1], c=y, cmap='tab10')
            plt.colorbar()
            # plt.scatter(tsne_v[:, 0], tsne_v[:, 1], c=y, cmap=cm.bwr)
            # cbar = fig.colorbar(colormapping, ax=plt.gca())
            plt.show()

        else:
            # vmin = tsne_v.max()
            # vmax = tsne_v.min()
            # norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            # colormapping = cm.ScalarMappable(norm=norm, cmap = cm.bwr)
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(tsne_v[:, 0], tsne_v[:, 1], tsne_v[:, 2], c=y, cmap='tab10')
            # ax.scatter(tsne_v[:, 0], tsne_v[:, 1], tsne_v[:, 2], c=y, cmap=cm.bwr)
            # plt.colorbar()
            plt.show()

    def cos_similarity_vis(self, target_input, target_label, cos_output_arr, cos_total_label, cos_similarity):

        if not os.path.exists("./cos_similarity_result"):
            os.makedirs("./cos_similarity_result")

        print(f"target label: {target_label}")
        print(f"total_label: {cos_total_label[:20]}")
        print(f"cos_similarity: {cos_similarity[:20]}")

        # plot parameters
        cols, rows = 5, 4
        fig2, axs2 = plt.subplots(rows, cols)
        inputs = []


        plt.imsave("./cos_similarity_result/target_input.png", np.array(target_input).reshape(28, 28), cmap='gray')

        for r in range(rows):
            for c in range(cols):
                axs2[r, c].set(yticklabels=[])
                axs2[r, c].set(xticklabels=[])
                axs2[r, c].tick_params(left=False)
                axs2[r, c].tick_params(right=False)
                axs2[r, c].tick_params(bottom=False)
                inputs.append(axs2[r, c].imshow(cos_output_arr[r+c].reshape(28, 28, 1), cmap='gray'))
                # axs[r, c].label_outer()

        fig2.savefig("./cos_similarity_result/cos_similarity_output_result.png")



    def evaluate(self):

        start = time.time()
        print(f"Start Evaluation! time: {start}")
        print(f"-"*10)

        test_loader = DataLoader(
            self.test_data, batch_size=self.args.batch_size, shuffle=False)

        valid_loader = DataLoader(
            self.valid_data, batch_size=self.args.batch_size, shuffle=False)

        self.test(test_loader, valid_loader)


    def test(self, test_loader, valid_loader):
        self.model.eval()
        total_label = []
        total_latent_vector = []
        output_arr = []
        input_arr = []

        for step, (x, label) in enumerate(test_loader):
            x = x.view(-1, self.args.height*self.args.width).to(self.device)
            label = label.to(self.device)

            output, latent_vector = self.model(x)

            input_arr.append(x)
            output_arr.append(output)
            total_label.append(label)
            total_latent_vector.append(latent_vector)

        # append에 넣고 numpy로 변경하는게 올바른 방법일까? 처리 시간의 효율을 높일 수 없을까?
        total_label = torch.cat(total_label, dim=0)
        total_latent_vector = torch.cat(total_latent_vector, dim=0)
        input_arr = torch.cat(input_arr, dim=0)
        output_arr = torch.cat(output_arr, dim=0)

        # cos 유사도를 비교할 target data를 validation data에서 하나 추출함.
        # target_output[0], target_lv[0] -> batch 중 첫번째 데이터를 사용
        target_data = next(iter(valid_loader))
        target_input = target_data[0]
        target_label = target_data[1]
        target_output, target_lv = self.model(target_input.view(-1, self.args.height*self.args.width).to(self.device))

        # test data 전체의 latent vector와 target latent vector의 cos similarity를 구한다.
        test_latent_vec = total_latent_vector
        target_latent_vec = target_lv[0]
        # 패키지 사용하면, cos_similarity = torch.nn.functional.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=2)
        cos_similarity = (torch.sum((test_latent_vec * target_latent_vec), 1)) / ((torch.sum(test_latent_vec**2, 1)**.5 * torch.sum(target_latent_vec**2)**.5))
        # output과 label 데이터를 cos similarity 오름차순 기준으로 정렬하기 위해서, 유사도 배열의 오름차순 정렬된 index 추출 (유사도가 높은 순서대로 정렬된 index를 뽑음)
        index = np.argsort(cos_similarity.cpu().detach().numpy())[::-1]
        cos_output_arr = torch.tensor((output_arr.cpu().detach().numpy()[index]))
        cos_total_label = torch.tensor((total_label.cpu().detach().numpy()[index]))

        # self.tsne(total_latent_vector, total_label, input_arr, output_arr)
        self.cos_similarity_vis(target_input[0], target_label[0], cos_output_arr, cos_total_label, np.sort(cos_similarity.cpu().detach().numpy())[::-1])

