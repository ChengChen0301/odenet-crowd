import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from random import randint
import random
import shutil
import cv2
from matplotlib import gridspec

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--viz', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
parser.add_argument("--mode", default='client')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(10)

left = -5.0
right = 5.0
up = 5.0
down = -5.0
width = 1.0
r = 0.3
door = np.array([right, 0])
wall = np.array([[right, up, left, up],  # 上
                 [left, up, left, down],  # 左
                 [left, down, right, down],  # 下
                 [right, down, door[0], -width],
                 [door[0], width, right, up]])

Np = 20
Nknn = 10


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()
        self.nfe = 0

        self.net1 = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

        self.net2 = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        self.net3 = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        self.net1.load_state_dict(torch.load(saveDir + '/net1_parameter.pkl'))
        self.net2.load_state_dict(torch.load(saveDir + '/net2_parameter.pkl'))
        self.net3.load_state_dict(torch.load(saveDir + '/net3_parameter.pkl'))

    def forward(self, t, y):
        self.nfe += 1
        y = y.float()
        theX = y[:, 0:2]
        theV = y[:, 2:4]
        ux = y[:, 4:2*Np+2]
        dxdt = theV

        unit_e = 1.0*(torch.tensor([right+0.6, 0]).to(device) - theX) * (1.0 / torch.sqrt(torch.sum(torch.square(
            theX - torch.tensor([right+0.6, 0]).to(device)), dim=1)).reshape(-1, 1))
        f_1 = self.net1(torch.cat([unit_e, theV], 1))

        rightdist = right - theX[:, 0].reshape(-1, 1)
        todoor = np.where((theX[:, 1].cpu() < width) & (theX[:, 1].cpu() > -width))[0]
        rightdist[todoor] = 100
        f_2 = self.net2(theX[:, 0].reshape(-1, 1) - left) * torch.tensor([[1, 0]]).to(device) \
              + self.net2(theX[:, 1].reshape(-1, 1) - down) * torch.tensor([[0, 1]]).to(device) \
              + self.net2(up - theX[:, 1].reshape(-1, 1)) * torch.tensor([[0, -1]]).to(device) \
              + self.net2(rightdist) * torch.tensor([[-1, 0]]).to(device)

        f_3 = torch.zeros((len(y), 2)).to(device)
        for j in range(Nknn):
            d_ij = torch.sqrt(torch.sum(torch.square(ux[:, 2 * j:2 * j + 2]), dim=1))
            f_3 += self.net3(d_ij.reshape(-1, 1)) * (-ux[:, 2 * j:2 * j + 2] / d_ij.reshape(-1, 1))

        dvdt = (f_1 + f_2 + f_3) / 80.0
        dudt = torch.zeros((len(y), 2*Nknn)).to(device)
        result = torch.cat([dxdt, dvdt], 1)
        result = torch.cat([result, dudt], 1)
        indices = np.where(theX[:, 0] > right)[0]
        result[indices, :] = 0.0
        return result


def compute_next(func, s, ux):
    in_put = torch.cat([s, ux], 1)
    out_put = odeint(func, in_put, batch_t)
    s_ = out_put[1, :, 0:4]
    return s_


def compute_trajectory(func, s0, Nt):
    # s0: x1, v1, ux1
    #     x2, v2, ux2
    #     xn, vn, uxn
    s = s0
    tt = 0
    outNp = 0
    saveData = s0[:, 0:4].unsqueeze(0)
    while outNp < Np and tt < Nt:
        outNp = 0
        s_ = compute_next(func, s[:, 0:4], s[:, 4:2*Nknn+4])
        for i in range(Np):
            theX = s_[i, 0:2].reshape(1, -1)
            if theX[:, 0] > right:
                outNp = outNp + 1
            restX = s_[torch.arange(s_.size(0)) != i, 0:2]
            pickindices = np.where(restX.reshape(-1, 2)[:, 0] < right)[0]
            restX = restX.reshape(-1, 2)[pickindices]
            ux = restX.reshape(1, -1) - theX.repeat(1, len(restX))
            length = np.shape(ux)[1]
            if length < 2 * Nknn:
                restU = 10.0 * torch.ones(2*Nknn - length).reshape(1, -1)
                ux = torch.cat([ux, restU], 1)
            dist = ux.reshape(-1, 2)
            dist = torch.sum(torch.square(dist), axis=1)
            _, indices = torch.sort(dist)
            indices = indices[0:Nknn]
            ux = ux.reshape(-1, 2)[indices]
            ux = ux.reshape(1, -1)
            if i == 0:
                allux = ux
            else:
                allux = torch.cat([allux, ux], 0)
        s = torch.cat([s_, allux], 1)
        saveData = torch.cat([saveData, s[:, 0:4].unsqueeze(0)], dim=0)
        tt = tt + 1
    return saveData


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def visualize(true_y, pred_y, t1, t2, itr):

    if args.viz:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        for j in range(len(wall)):
            ax_traj.plot([wall[j][0], wall[j][2]], [wall[j][1], wall[j][3]], 'black')

        for i in range(Np-1):
            ax_traj.plot(true_y.cpu().numpy()[:, 2*i], true_y.cpu().numpy()[:, 2*i+1], color=colors[i], ls='-')
            ax_traj.plot(pred_y.cpu().numpy()[:, i, 0], pred_y.cpu().numpy()[:, i, 1], color=colors[i], ls='--')
        i = Np-1
        ax_traj.plot(true_y.cpu().numpy()[:, 2 * i], true_y.cpu().numpy()[:, 2 * i + 1], color=colors[i], ls='-', label='ORCA')
        ax_traj.plot(pred_y.cpu().numpy()[:, i, 0], pred_y.cpu().numpy()[:, i, 1], color=colors[i], ls='--', label='ODE-Net')

        ax_traj.set_xlim(left, right)
        ax_traj.set_ylim(down, up)
        ax_traj.set_xticks([])
        ax_traj.set_yticks([])
        ax_traj.set_aspect(1)
        ax_traj.legend()

        ax_dist.cla()
        ax_dist.set_title('Distance to the exit')
        for i in range(Np - 1):
            count = np.where(true_y.cpu().numpy()[:, 2*i] > 5.0)
            Nt = count[0][0] + 1
            dist_true = np.sqrt(np.square(true_y.cpu().numpy()[:len(t1), 2 * i] - right) + np.square(
                true_y.cpu().numpy()[:len(t1), 2 * i + 1]))
            dist_true[Nt:] = dist_true[Nt-1]
            ax_dist.plot(t1.cpu().numpy(), dist_true, color=colors[i], ls='-')
            dist_pred = np.sqrt(np.square(pred_y.cpu().numpy()[:len(t2), i, 0] - right) + np.square(
                pred_y.cpu().numpy()[:len(t2), i, 1]))
            ax_dist.plot(t2.cpu().numpy(), dist_pred, color=colors[i], ls='--')
        i = Np - 1
        count = np.where(true_y.cpu().numpy()[:, 2 * i] > 5.0)
        Nt = count[0][0] + 1
        dist_true = np.sqrt(
            np.square(true_y.cpu().numpy()[:len(t1), 2 * i] - right) + np.square(
                true_y.cpu().numpy()[:len(t1), 2 * i + 1]))
        dist_true[Nt:] = dist_true[Nt - 1]
        ax_dist.plot(t1.cpu().numpy(), dist_true, color=colors[i], ls='-', label='ORCA')
        dist_pred = np.sqrt(
            np.square(pred_y.cpu().numpy()[:len(t2), i, 0] - right) + np.square(pred_y.cpu().numpy()[:len(t2), i, 1]))
        ax_dist.plot(t2.cpu().numpy(), dist_pred, color=colors[i], ls='--', label='ODE-Net')

        ax_dist.set_aspect(0.9)
        ax_dist.legend()
        ax_dist.set_xlabel('t(s)')
        ax_dist.set_ylabel('distance(m)')

        fig.tight_layout()
        plt.subplots_adjust(wspace=-0.2)
        plt.savefig(saveDir + '/png_all20_2/{:03d}.eps'.format(itr), format='eps', bbox_inches='tight', pad_inches=0.1)
        plt.draw()
        plt.pause(0.001)


def showprocess(true_y, pred_y, tt):

    ax_pred.cla()
    for j in range(len(wall)):
        ax_pred.plot([wall[j][0], wall[j][2]], [wall[j][1], wall[j][3]], 'black')

    for i in range(Np):
        if pred_y[tt, i, 0] + r < right:
            circle = mpatches.Circle(xy=(pred_y[tt, i, 0], pred_y[tt, i, 1]), radius=r, color=colors[i], alpha=1)
            ax_pred.add_patch(circle)
    ax_pred.set_aspect(1)
    ax_pred.set_title("ODE-Net")
    ax_pred.set_xticks([])
    ax_pred.set_yticks([])

    ax_true.cla()
    for j in range(len(wall)):
        ax_true.plot([wall[j][0], wall[j][2]], [wall[j][1], wall[j][3]], 'black')

    for i in range(Np):
        if true_y[tt, 2 * i] + r < right:
            circle = mpatches.Circle(xy=(true_y[tt, 2 * i], true_y[tt, 2 * i + 1]), radius=r, color=colors[i], alpha=1)
            ax_true.add_patch(circle)
    ax_true.set_aspect(1)
    ax_true.set_title("ORCA")
    ax_true.set_xticks([])
    ax_true.set_yticks([])

    plt.suptitle("$t = $" + str(tt / 100.0) + " $s$", y=0.78, fontsize=16)
    plt.draw()
    plt.savefig(saveDir + '/png_all20/' + str(tt) + '.png', bbox_inches='tight', pad_inches=0.1)

    plt.pause(0.001)


def imgs2video(length):

    fps = 10
    size = (1589, 962)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    imgs_dir = saveDir + '/png20/'
    save_name = saveDir + '/video20.avi'
    video = cv2.VideoWriter(save_name, fourcc, fps, size)

    i = 0
    while i < length:
        filepath = imgs_dir + str(i) + '.png'
        img = cv2.imread(filepath)
        video.write(img)
        i = i+10  # 0.5 seconds

    video.release()


if __name__ == "__main__":

    saveDir = "result/orca"
    makedirs(saveDir)

    func = ODEFunc().to(device)
    batch_t = torch.tensor([0, 0.01]).to(device)

    colors = []
    for i in range(Np):
        colors.append('#%06X' % randint(0, 0xFFFFFF))

    if args.viz:
        fig = plt.figure(figsize=(10, 4))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 3])
        ax_traj = plt.subplot(gs[0], frameon=False)
        ax_dist = plt.subplot(gs[1], frameon=True)
        # fig = plt.figure(figsize=(10, 10), dpi=100)
        # ax_true = fig.add_subplot(121, frameon=False)
        # ax_pred = fig.add_subplot(122, frameon=False)
        plt.show(block=False)

    numbers1 = [[]]
    numbers2 = [[]]
    ev1 = []
    ev2 = []

    for k in range(1):
        k = 199
        # print(k)
        dir = 'RVO2/examples/orca_20/round'
        folder = dir + str(k + 1)
        for i in range(2000):
            filename = folder + '/velocity' + str(i) + '.txt'
            if os.path.exists(filename):
                vel = np.loadtxt(folder + '/velocity' + str(i) + '.txt')
                coord = np.loadtxt(folder + '/coord' + str(i) + '.txt')
                if i == 0:
                    totalvel = vel
                    totalcoord = coord
                else:
                    totalvel = np.concatenate((totalvel, vel), 0)
                    totalcoord = np.concatenate((totalcoord, coord), 0)
            else:
                break

        coord = totalcoord[0:Np]
        vel = totalvel[0:Np]
        coord = torch.tensor(coord)
        vel = torch.tensor(vel)
        totalcoord = totalcoord.reshape(-1, Np, 2)
        totalvel = totalvel.reshape(-1, Np, 2)
        totalcoord = totalcoord.reshape(-1, 2 * Np)
        totalvel = totalvel.reshape(-1, 2 * Np)
        data = np.concatenate((totalcoord, totalvel), 1)
        data = torch.tensor(data)

        for j in range(Np):
            s = torch.cat([coord[j].reshape(1, -1), vel[j].reshape(1, -1)], 1)
            if j == 0:
                restX = coord[j + 1:Np, :]
            elif j == Np - 1:
                restX = coord[0:j, :]
            else:
                restX = torch.cat([coord[0:j, :], coord[j + 1:Np, :]], 0)

            pickindices = np.where(restX.reshape(-1, 2)[:, 0] < right)[0]
            restX = restX.reshape(-1, 2)[pickindices]
            ux = restX.reshape(1, -1) - coord[j].reshape(1, -1).repeat(1, len(restX))
            length = np.shape(ux)[1]
            if length < 2 * Nknn:
                restU = 10.0 * torch.ones(2 * Nknn - length).reshape(1, -1)
                ux = torch.cat([ux, restU], 1)
            dist = ux.reshape(-1, 2)
            dist = torch.sum(torch.square(dist), axis=1)
            _, indices = torch.sort(dist)
            indices = indices[0:Nknn]
            ux = ux.reshape(-1, 2)[indices]
            ux = ux.reshape(1, -1)
            if j == 0:
                alls = s
                allux = ux
            else:
                alls = torch.cat([alls, s], 0)
                allux = torch.cat([allux, ux], 0)
        s0 = torch.cat([alls, allux], 1)
        with torch.no_grad():
            saveData = compute_trajectory(func, s0, 2000)
            t2 = torch.linspace(0., (len(saveData) - 1) / 100, len(saveData))
            t1 = torch.linspace(0., (len(data) - 1) / 100, len(data))
            visualize(data, saveData, t1, t2, k)

        # if os.path.exists(saveDir + '/png20_2'):
        #     shutil.rmtree(saveDir + '/png20_2')
        # makedirs(saveDir + '/png20_2')
        # showprocess(data, saveData, 0)
        # for tt in range(2000):
        #     if tt % 10 == 0:
        #         showprocess(data, saveData, tt)
        # imgs2video(1403)

    #         ev1.append((len(data) + 1) / 100)
    #         ev2.append(len(saveData) / 100)
    #         out1 = []
    #         out2 = []
    #         for i in range(max(len(data)+1, len(saveData))):
    #             if i < len(data):
    #                 # out1.append(len(np.where(data[i, [0, 2, 4, 6, 8]] < right)[0]))
    #                 out1.append(len(np.where(data[i, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38]] < right)[0]))
    #             else:
    #                 out1.append(0)
    #             if i < len(saveData):
    #                 out2.append(len(np.where(saveData[i, :, 0] < right)[0]))
    #             else:
    #                 out2.append(0)
    #         if k == 0:
    #             numbers1[0] = out1
    #             numbers2[0] = out2
    #         else:
    #             numbers1.append(out1)
    #             numbers2.append(out2)
    #
    # max_len = max((len(l) for l in numbers1))  # round条曲线的最大长度
    # new_numbers = list(map(lambda l: l + [0] * (max_len - len(l)), numbers1))
    # array = np.array(new_numbers)
    # aver_num1 = array.sum(axis=0) / 200
    # max_len = max((len(l) for l in numbers2))  # round条曲线的最大长度
    # new_numbers = list(map(lambda l: l + [0] * (max_len - len(l)), numbers2))
    # array = np.array(new_numbers)
    # aver_num2 = array.sum(axis=0) / 200
    # np.save(saveDir + '/aver_num1_2.npy', aver_num1)
    # np.save(saveDir + '/aver_num2_2.npy', aver_num2)
    # np.save(saveDir + '/ev1_2.npy', ev1)
    # np.save(saveDir + '/ev2_2.npy', ev2)

    # aver_num1 = np.load(saveDir + '/aver_num1_3.npy')
    # aver_num2 = np.load(saveDir + '/aver_num2_3.npy')
    # fig = plt.figure(figsize=(12, 5))
    # ax1 = fig.add_subplot(121, frameon=True)
    # ax2 = fig.add_subplot(122, frameon=True)
    # X1 = np.linspace(0, len(aver_num1) / 100, num=len(aver_num1), endpoint=False, retstep=False, dtype=None)
    # ax1.plot(X1, (Np - aver_num1) / Np, ls='-', label='ORCA')
    # ax1.plot(X1, (Np - aver_num2) / Np, ls='--', label='ODE-Net')
    # ax1.set_xlabel("t (s)")
    # ax1.set_ylabel("$N_{out} / N$")
    # ax1.legend()
    # # plt.savefig(saveDir + "/png_all_20/averOutNum_for_" + str(200) + "_rounds.png", bbox_inches='tight', pad_inches=0.1)
    # ev1 = np.load(saveDir + '/ev1_3.npy')
    # ev2 = np.load(saveDir + '/ev2_3.npy')
    # bins = np.linspace(9, 16, 15)
    # ax2.hist([ev1, ev2], bins, density=True, label=['ORCA', 'ODE-Net'])
    # ax2.legend(loc='upper right')
    # ax2.set_xlabel("$T_{ev}$ (s)")
    # ax2.set_ylabel("Probability")
    # # plt.hist(ev1, 30, density=True, facecolor='blue', alpha=0.5)
    # # plt.hist(ev2, 30, density=True, facecolor='red', alpha=0.5)
    # plt.savefig(saveDir + "/png_all20/hist_for_" + str(200) + "_rounds3.eps", format='eps', bbox_inches='tight', pad_inches=0.1)
