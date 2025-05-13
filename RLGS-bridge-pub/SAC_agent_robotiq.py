import numpy as np
import torch
import torch.nn as nn
import pickle
import random
import itertools
import torch.nn.functional as F
from utils.aug_utils import random_color_jitter
import cv2 
import os

count_suc = 0
def xdist(pos, orn, x_n):
    #print()
    #return pos[0] + 0.005 * np.cos(orn[2] - np.pi * 3 / 4) - x_n
    return pos[0] - x_n


def ydist(pos, orn, y_n):
    #return pos[1] + 0.005 * np.sin(orn[2] - np.pi * 3 / 4) - y_n
    return pos[1]  - y_n


def da(angle, orn, K=1.5):
    return np.tanh(K * (angle - orn))
    

def dang(angle, orn, K=1.5):
    #return np.tanh(2 * (angle - orn))
    if (angle - orn) < -np.pi:
        delta = angle - orn + 2*np.pi
    elif (angle - orn) > np.pi:
        delta = angle - orn - 2*np.pi
    else:
        delta = angle - orn
    return np.tanh(K * delta)

def diff_max(A, B):
        with torch.no_grad():
            xi = nn.ReLU()(torch.sign(A - B))
        return A*xi + B*(1-xi)

def diff_min(A, B):
    with torch.no_grad():
        xi = nn.ReLU()(torch.sign(B - A))
    return A*xi + B*(1-xi)

class BaseController:
    def __init__(self, z1, z2, z3, z4, K = 5, ang = -0.2, K2=5, Kz2 = 5, refine=False, force=0.0, rot_90=False, dist=0.02, height = 0.09, place=False, bias=0):
        self.count_suc = 0
        self.count_grip = 0
        self.count_force = 0
        self.z1 = z1
        self.z2 = z2
        self.z3 = z3
        self.z4 = z4
        self.K = K
        self.Kz = K - 2
        self.open_ang = ang
        self.K2 = K2
        self.Kz2 = Kz2
        self.refine = refine
        self.force = force
        self.rot_90 = rot_90
        self.dist = dist
        self.height = height
        self.place = place
        self.bias = bias
    def act(self, s):
        if self.refine:
            #rand1 = (np.random.rand()-0.5)*0.03
            #rand2 = (np.random.rand()-0.5)*0.03
            height1 = self.z1 + (np.random.rand()-0.5)*0.02 - 0.01
            height2 = self.z2 + (np.random.rand()-0.5)*0.02
            height3 = self.z3 + (np.random.rand()-0.5)*0.05
            height4 = self.z4 + np.random.rand()*0.02 -0.01
        else:
            height1 = self.z1
            height2 = self.z2
            height3 = self.z3
            height4 = self.z4
        return self.base_template(s, height1, height2, height3, height4)
    def base_template(self, s, z1, z2, z3, z4):
        ### z1 grasping height
        ### z2: gripper open height
        ### z3: lift height
        ### z4: place height
        ### 因为爪的原点不是最底下，所以z都比较高一点呢
        x_n = s[0]
        y_n = s[1]
        height = s[2]
        gripper_angle = s[5]
        finger_angle = s[6]
        #print("angle:", finger_angle)
        finger_force = s[7]
        
        pos1 = s[8:11]
        orn1 = s[11:14]
        if self.rot_90:
            orn1[2] = orn1[2]+np.pi/2
        if self.bias:
            pos1[0] = pos1[0] + self.bias * np.sin(orn1[2])
            pos1[1] = pos1[1] - self.bias * np.cos(orn1[2])
        pos2 = s[14:17]
        orn2 = s[17:20]
        #orn2 = [0, 0, 0]
        if self.rot_90:
            orn2[2] = orn2[2] + np.pi / 2
        if self.place:
            pos2 = [0.4, 0, 0]
            orn2 = [0, 0, 0]
        action = np.zeros(5)
        #print("dist:", np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2))
        if np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2) < self.dist and pos1[2] < self.height:
            #print('in')
            self.count_suc += 1
        else:
            self.count_suc = 0
        
        #print('finger_force', finger_force)
        if finger_force < 1:
            if np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2) < self.dist and pos1[2] < self.height and self.count_suc > 3:
                #### 复位
                action[2] = np.tanh(self.K * (z3 - pos1[2]))
                self.count_force = 0
            else:
                #print("in gripping")
                ##### 夹取物块
                action[0] = np.tanh(xdist(pos1, orn1, x_n) * self.K)
                action[1] = np.tanh(ydist(pos1, orn1, y_n) * self.K)
                action[2] = np.tanh(self.Kz * (pos1[2] + z1 - height))
                #action[2] = 0
                action[3] = dang(gripper_angle, orn1[2])
                action[4] = da(self.open_ang, finger_angle)
                if pos1[2] + z1 < height < z2:
                    #print("to gripping")
                    #action[4] = da(0.1, finger_angle)
                    action[4] = da(-0.5, finger_angle, 2)
                    action[2] = np.tanh(self.Kz * (pos1[2] + z1 + 0.02 - height))
                    self.count_force += 1
                    #print('num:', finger_angle, action[4])
                if self.count_force > 5:
                    self.count_force = 0
            self.count_grip = 0
        elif self.count_force < 5:
            if pos1[2] + z1 < height < z2:
                action[0] = 0
                action[1] = 0
                action[2] = 0
                action[3] = 0
                action[4] = da(-0.6, finger_angle, K=2)
            self.count_force += 1
        else:
            #print("in setting")
            ##### place the object
            action[0] = np.tanh((pos2[0] - pos1[0]) * self.K2)
            action[1] = np.tanh((pos2[1] - pos1[1]) * self.K2)
            action[2] = np.tanh(self.Kz2 * (z3 - pos1[2]))
            action[3] = dang(gripper_angle, orn2[2])
            action[4] = da(-0.6, finger_angle, K=2)

            #### open the gripper
            #count = 0
            if np.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2) < self.dist:
                #print("in songzhua")
                self.count_grip += 1

                action[2] = np.tanh(5 * (z4 - pos1[2]))
                if self.place:
                    count = 5
                else:
                    count = 15
                if pos1[2] < z4 + 0.05 and self.count_grip > count:
                    action[4] = da(self.force, finger_angle)
                    #self.count_suc = 6
            else:
                self.count_grip = 0
            #print('suc_count:', self.count_suc)
        
        return action


class Base2(BaseController):
    def __init__(self, refine = False):
        super().__init__(0.03, 0.24, 0.2, 0.090, K2=3, Kz2=6, refine=refine, dist=0.015, height=0.090)
    #
    #return base_template(s, z1=height1, z2=height2, z3=0.15, z4=height4)

class Base3(BaseController):
    def __init__(self, refine = False):
        super().__init__(0.03, 0.24, 0.15, 0.03, K2=3, refine=refine, place=True, height=0.050, dist=0.045)

class Base4(BaseController):
    def __init__(self, refine = False):
        ### cake and small_cube and banana
        super().__init__(0.03, 0.24, 0.20, 0.20, K2=0, refine=refine, force=-0.6)
        ### bear
        #super().__init__(0.01, 0.24, 0.20, 0.20, K2=0, ang=0, refine=refine, force=-0.6, bias=0.001)


def base3_ensemble(info):
    x_n = info[0]
    y_n = info[1]
    height = info[2]
    gripper_angle = info[5]
    finger_angle = info[6]
    finger_force = info[7]
    cups_pos = info[8:11]
    cups_orn = info[11:14]
    cup_pos = info[14:17]
    cup_orn = info[17:20]
    def b0():
        action = np.zeros(5)
        if finger_force < 1:
            action[0] = np.tanh(xdist(cups_pos, cups_orn, x_n) * 5)
            action[1] = np.tanh(ydist(cups_pos, cups_orn, y_n) * 5)
            action[2] = np.tanh(5 * (cups_pos[2] + 0.3 - height))
            action[3] = da(gripper_angle, cups_orn[2])
            action[4] = da(0, finger_angle)
            if cups_pos[2] + 0.3 < height < 0.45:
                action[4] = da(0.2, finger_angle)
        else:
            action[0] = np.tanh((cup_pos[0] - cups_pos[0]) * 5)
            action[1] = np.tanh((cup_pos[1] - cups_pos[1]) * 5)
            action[2] = np.tanh(5 * (0.35 - cups_pos[2]))
            action[3] = da(gripper_angle, cup_orn[2])
            action[4] = da(0, finger_angle)
            if np.sqrt((cup_pos[0] - cups_pos[0]) ** 2 + (cup_pos[1] - cups_pos[1]) ** 2) < 0.02:
                action[4] = da(0.2, finger_angle)
        return action


    def b1():
        action = np.zeros(5)
        if finger_force < 1:
            if xdist(cups_pos, cups_orn, x_n) < 0.05 and ydist(cups_pos, cups_orn, y_n) < 0.05:
                action[0] = np.tanh(xdist(cups_pos, cups_orn, x_n) * 5)
                action[1] = np.tanh(ydist(cups_pos, cups_orn, y_n) * 5)
                action[2] = np.tanh(5 * (cups_pos[2] + 0.3 - height))
                action[3] = da(gripper_angle, cups_orn[2])
                action[4] = da(0, finger_angle)
                if cups_pos[2] + 0.3 < height < 0.45:
                    action[4] = da(0.2, finger_angle)
            else:
                action[0] = np.tanh(xdist(cups_pos, cups_orn, x_n) * 5)
                action[1] = np.tanh(ydist(cups_pos, cups_orn, y_n) * 5)
                action[2] = 0
                action[3] = da(gripper_angle, cups_orn[2])
                action[4] = da(0, finger_angle)
                if cups_pos[2] + 0.3 < height < 0.45:
                    action[4] = da(0.2, finger_angle)
        else:
            if np.sqrt((cup_pos[0] - cups_pos[0]) ** 2 + (cup_pos[1] - cups_pos[1]) ** 2) > 0.05 and cups_pos[2] < 0.25:
                action[0] = 0
                action[1] = 0
                action[2] = np.tanh(5 * (0.35 - cups_pos[2]))
                action[3] = da(gripper_angle, cup_orn[2])
                action[4] = da(0, finger_angle)

            else:
                action[0] = np.tanh((cup_pos[0] - cups_pos[0]) * 5)
                action[1] = np.tanh((cup_pos[1] - cups_pos[1]) * 5)
                action[2] = np.tanh(5 * (0.35 - cups_pos[2]))
                action[3] = da(gripper_angle, cup_orn[2])
                action[4] = da(0, finger_angle)
            if np.sqrt((cup_pos[0] - cups_pos[0]) ** 2 + (cup_pos[1] - cups_pos[1]) ** 2) < 0.02:
                action[2] = np.tanh(5 * (0.15 - cups_pos[2]))
                if cups_pos[2] < 0.2:
                    action[4] = da(0.2, finger_angle)
        return action
    return b0(),b1()


def base_controller(s, base):
    size = s.shape[0]
    action = np.zeros((size, 5), dtype=np.float32)
    for i in range(size):
        action[i, :] = base.act(s[i, :])
    return action


def base_controller_ensemble(s, base):
    size = s.shape[0]
    action1 = np.zeros((size, 5), dtype=np.float32)
    action2 = np.zeros((size, 5), dtype=np.float32)
    action3 = np.zeros((size, 5), dtype=np.float32)
    for i in range(size):
        action1[i, :], action2[i, :], action3[i, :] = base(s[i, :])
    return action1, action2, action3


def opt_cuda(t, device):
    if torch.cuda.is_available():
        cuda = "cuda:" + str(device)
        return t.cuda(cuda)
    else:
        return t


def np_to_tensor(n, device):
    return opt_cuda(torch.from_numpy(n).type(torch.FloatTensor), device)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class SpacialSoftmaxExpectation(nn.Module):
    def __init__(self, size, device):
        super(SpacialSoftmaxExpectation, self).__init__()
        cor = opt_cuda(torch.arange(size).type(torch.FloatTensor), device)
        X, Y = torch.meshgrid(cor, cor)
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        self.fixed_weight = torch.cat((Y, X), dim=1)
        self.fixed_weight /= size - 1

    def forward(self, x):
        
        return nn.Softmax(2)(x.view(*x.size()[:2], -1)).matmul(self.fixed_weight).view(x.size(0), -1)

def gaussian_logprob(noise, log_std):
	"""Compute Gaussian log probability"""
	residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
	return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
	"""Apply squashing function, see appendix C from https://arxiv.org/pdf/1812.05905.pdf"""
	mu = torch.tanh(mu)
	if pi is not None:
		pi = torch.tanh(pi)
	if log_pi is not None:
		log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
	return mu, pi, log_pi

class Actor(nn.Module):
    def __init__(self, mode, width, device, use_force=True):
        super(Actor, self).__init__()
        self.mode = mode
        self.width = width
        self.device = device
        self.log_std_min = -10
        self.log_std_max = 2
        self.use_force = use_force
        #self.in_channel = 4 if self.mode == 'rgbd' else 3
        #self.out_channel = 16 if self.mode == 'rgbd' else 8
        if self.mode == 'rgbd':
            self.in_channel = 4 
            self.out_channel = 16 
        elif self.mode == 'de':
            self.in_channel = 3
            self.out_channel = 8
        else:
            self.in_channel = 3
            self.out_channel = 16
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_channel, self.out_channel, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channel, self.out_channel, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channel, self.out_channel, 3, 1),
            nn.ReLU(inplace=True))
        if self.use_force:
            self.fc = nn.Sequential(
                nn.Linear(32 + 8, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 5*2),
                #Tanh
                )
        else:
            self.fc = nn.Sequential(
                nn.Linear(32 + 7, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 5*2),
                #Tanh
                )
        
        # from torchsummary import summary

        # summary(self.conv.to(device),input_size=(3,128,128))

    def forward(self, x, robot_state, compute_pi=True, compute_log_pi=True, border_act=False):
        if self.mode == 'rgbd':
            x2 = self.conv(x / 255)
        elif self.mode == 'de':
            x2 = torch.cat((self.conv(x[:, :3] / 255), self.conv(x[:, 3:] / 255)), dim=1)
        else:
            #print("x shape:", x[:, :3].shape)
            x2 = self.conv(x[:, :3] / 255)
            #### vis feat #######
            # from test_utils import vis_feat
            # feat = x2.detach().cpu().reshape((x2.shape[0], x2.shape[1], -1)).transpose(1,2)#.numpy()
            # patch_w = int(self.width-6)
            # path = 'train_image_CNN.png'
            # vis_feat(feat, path, patch_w)
        x3 = SpacialSoftmaxExpectation(self.width - 6, self.device)(x2)
        # concatenate with robot state:
        if self.use_force:
            mu, log_std = self.fc(torch.cat((x3, robot_state), dim=1)).chunk(2, dim=-1)
        else:
            robot_state = robot_state[:, :7]
            mu, log_std = self.fc(torch.cat((x3, robot_state), dim=1)).chunk(2, dim=-1)
        #print("a:", a)
        #mu, log_std = self.layers(x).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None

        if border_act: ## better than 60% act
            std = log_std.exp()
            border_pi1 = mu + 0.53 * std #0.53 60% 0.26 80%
            border_pi2 = mu - 0.53 * std

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)
		
        if border_act: 
            return mu, pi, log_pi, log_std, border_pi1, border_pi2
        else:
            return mu, pi, log_pi, log_std


class FastActor(nn.Module):
    def __init__(self):
        super(FastActor, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(20, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 5),
            nn.Tanh())

    def forward(self, s):
        return self.fc(s)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(20 + 5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid())

        self.fc2 = nn.Sequential(
            nn.Linear(20 + 5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid())

    def forward(self, state, action):
        return self.fc1(torch.cat((state, action), dim=1)), self.fc2(torch.cat((state, action), dim=1))


class ReplayBuffer:
    def __init__(self, c, w, state_dim, action_dim, size):
        self.sta1_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.sta2_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.obv1_buf = np.zeros([size, c, w, w], dtype=np.uint8)
        self.obv2_buf = np.zeros([size, c, w, w], dtype=np.uint8)
        self.acts_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        self.done_buf = np.zeros([size, 1], dtype=np.bool_)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obv, sta, act, next_obv, next_sta, rew, done):
        self.sta1_buf[self.ptr] = sta
        self.sta2_buf[self.ptr] = next_sta
        self.obv1_buf[self.ptr] = obv
        self.obv2_buf[self.ptr] = next_obv
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(sta1=self.sta1_buf[idxs],
                    sta2=self.sta2_buf[idxs],
                    obv1=self.obv1_buf[idxs],
                    obv2=self.obv2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


class ReplayBufferFast:
    def __init__(self, state_dim, action_dim, size):
        self.sta1_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.sta2_buf = np.zeros([size, state_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size, 1], dtype=np.float32)
        self.done_buf = np.zeros([size, 1], dtype=np.bool_)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, sta, act, next_sta, rew, done):
        self.sta1_buf[self.ptr] = sta
        self.sta2_buf[self.ptr] = next_sta
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def merge(self, buffer_fast):
        for i in range(buffer_fast.size):
            self.store(buffer_fast.sta1_buf[i],
                       buffer_fast.acts_buf[i],
                       buffer_fast.sta2_buf[i],
                       buffer_fast.rews_buf[i],
                       buffer_fast.done_buf[i])

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(sta1=self.sta1_buf[idxs],
                    sta2=self.sta2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])



class SACWBAgent(object):
    def __init__(self, log_dir, mode='de', width=128, device=0, use_fast=False, task=1, 
                mixed_q=True, base_boot=True, behavior_clone=True, refine = False, pretrain = False, color_jitter = False, Qb_strain = False, use_force = True):

        self.mode = mode
        self.width = width
        self.device = device
        self.use_fast = use_fast
        self.task = task
        self.mixed_q = mixed_q
        self.base_boot = base_boot
        self.behavior_clone = behavior_clone
        self.pretrain = pretrain
        self.color_jitter = color_jitter
        self.Qb_strain = Qb_strain #Qb_strain
        self.Qb_strain_critic_maxQ = False
        self.Qb_strain_actor = Qb_strain #Qb_strain
        self.Qb_strain_critic = False #Qb_strain ### TODO 改回来！
        self.Qb_strain_max = False #这个不要动
        self.use_mu = True#Qb_strain
        if task==4:
            self.frame_num = 17e4
        else:
            self.frame_num = 35e4

        # if self.pretrain:
        #     self.frame_num = self.frame_num - 1e4
        
        if self.task == 1:
            #self.base = base1
            #base = base1
            raise NotImplementedError("Task 1 not implement yet!")
        elif self.task == 2:
            self.base = Base2(refine=refine)
        elif self.task == 3:
            self.base = Base3(refine=refine)
        elif self.task == 4:
            self.base = Base4(refine=refine)
        if self.use_fast:
            self.buffer = ReplayBufferFast(20, 5, size=1000000)
        else:
            self.buffer = ReplayBuffer(6 if (self.mode == 'de' or self.mode == 'mono') else 4, self.width, 20, 5, size=100000)
        if self.use_fast:
            self.actor = opt_cuda(FastActor(), self.device)
        else:
            print("using force:", use_force)
            self.actor = opt_cuda(Actor(mode=self.mode, width=self.width, device=self.device, use_force=use_force), self.device)
        self.critic = opt_cuda(Critic(), self.device)
        self.target_critic = opt_cuda(Critic(), self.device)
        if self.pretrain:
            print("pretrain:")
            self.actor.load_state_dict(torch.load("/data/neural-scene-graphs-3dgs/saves/SAC_t2i_eih/gift_SAC_0902_2r/actor_best.pt").state_dict(), strict=False)
        
        soft_update(self.target_critic, self.critic, 1)
        
        self.gamma = 0.994
        self.tau = 0.005
        self.epsilon = 1
        # if self.pretrain:
        #     self.epsilon = 0.1
        self.delta = 1e-5 #wyx change:2e-5

        self.batch_size = 256

        #### TODO new 参数
        self.log_alpha = torch.tensor(np.log(0.1)).cuda()
        self.log_alpha.requires_grad = True
        #### TODO new 参数
        self.target_entropy = -np.prod([5,]) ## action shape
        #self.target_entropy = -np.prod([10,])# low entropy

        if self.self.Qb_strain_critic:
            self.log_lamb1 = torch.tensor(np.log(1)).cuda()
            self.log_lamb2 = torch.tensor(np.log(1)).cuda()
            self.log_lamb3 = torch.tensor(np.log(1)).cuda()
            self.log_lamb1.requires_grad = True
            self.log_lamb2.requires_grad = True
            self.log_lamb3.requires_grad = True

            if self.Qb_strain_max:
                self.log_lamb_optimizer = torch.optim.Adam([self.log_lamb1, self.log_lamb2, self.log_lamb3], lr=1e-4, betas=(0.5, 0.999))
            else:
                self.log_lamb_optimizer = torch.optim.Adam([self.log_lamb1, self.log_lamb2], lr=1e-4, betas=(0.5, 0.999))

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)  #raw 1e-3 3e-4
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3) #raw 1e-3
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-4, betas=(0.5, 0.999)) ### raw: 1e-4
        # self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        # self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        ### TODO:加吗?
        #self.aug = m.RandomShiftsAug(pad=4)

        ### TODO 这啥
        #self.train()

    def load_ckpt(self, actor, critic):
        self.actor = actor
        self.critic = critic
        self.target_critic = critic

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def lamb1(self):
        return self.log_lamb1.exp()

    @property
    def lamb2(self):
        return self.log_lamb2.exp()

    @property
    def lamb3(self):
        return self.log_lamb3.exp()

    def act(self, o, s, test=False):

        action_b = self.base.act(s)
        s = np_to_tensor(s, self.device).unsqueeze(dim=0)
        if self.use_fast:
            with torch.no_grad():
                ### TODO: state actor需要更新
                _, action, _, _ = self.actor(s, compute_log_pi=False)
        else:
            o = np_to_tensor(o, self.device).unsqueeze(dim=0)
            s_p = s[:, :8]
            with torch.no_grad():
                ### TODO: actor需要更新
                mu, action, _, _  = self.actor(o, s_p, compute_log_pi=False)
        if test:
            return mu.squeeze().cpu().numpy(), True
        if np.random.uniform(0, 1) < self.epsilon:
            self.epsilon = max(self.epsilon - self.delta, 0)
            return action_b, False
        else:
            self.epsilon = max(self.epsilon - self.delta, 0)
            action_b_t = np_to_tensor(action_b, self.device).unsqueeze(dim=0)
            with torch.no_grad():
                ### TODO: critic需要更新
                q_b1, q_b2 = self.critic(s, action_b_t)
                q_b = torch.min(q_b1, q_b2)
                q1, q2 = self.critic(s, action)
                q = torch.min(q1, q2)
            if q_b.item() > q.item() and self.mixed_q:
                return action_b, False
            else:
                return action.squeeze().cpu().numpy(), True

    def remember(self, observation, state, action, next_observation, next_state, reward, done):
        if self.use_fast:
            self.buffer.store(state, action, next_state, [reward], [done])
        else:
            self.buffer.store(observation, state, action, next_observation, next_state, [reward], [done])

    def update_critic(self, oi, si, ai, ri, on, sn, d, base_action_n=None, sn_p=None):

        with torch.no_grad():
            if self.use_fast:
                _, a_next, log_pi, _ = self.actor(sn)
            else:
                if self.Qb_strain_critic:
                    _, a_next, log_pi, _, border_act1, border_act2 = self.actor(on, sn_p, border_act=True)
                else:
                    _, a_next, log_pi, _ = self.actor(on, sn_p)
            back_up1, back_up2 = self.target_critic(sn, a_next)
            back_up = torch.min(back_up1, back_up2)
            if self.base_boot:
                back_up_d1, back_up_d2 = self.target_critic(sn, base_action_n)
                back_up_d = torch.min(back_up_d1, back_up_d2)
                if self.Qb_strain_critic_maxQ and np.random.uniform(0, 1) < self.epsilon:
                    back_up = back_up_d
                else:
                    back_up = torch.max(back_up, back_up_d)
            if self.Qb_strain_critic:
                back_up_d1, back_up_d2 = self.critic(sn, base_action_n)
                back_up_d = diff_min(back_up_d1, back_up_d2)
                if self.Qb_strain_max:
                    back_up_d_max = diff_max(back_up_d1, back_up_d2)
                # back_up_b11, back_up_b12 = self.critic(sn, border_act1)
                # back_up_b1 = diff_max(back_up_b11, back_up_b12)
                # back_up_b21, back_up_b22 = self.critic(sn, border_act2)
                # back_up_b2 = diff_max(back_up_b21, back_up_b22)

                back_up_b11, back_up_b12 = self.target_critic(sn, border_act1)
                back_up_b1 = torch.max(back_up_b11, back_up_b12)
                back_up_b21, back_up_b22 = self.target_critic(sn, border_act2)
                back_up_b2 = torch.max(back_up_b21, back_up_b22)

            target_V = back_up - self.alpha.detach() * log_pi
            #print(ri)
            # print("back_up:", back_up.mean())
            # print("target V:", target_V.mean())
            # print("gamma V:", self.gamma * target_V.mean())
            yi = ri + (1 - d) * self.gamma * target_V
        qi1, qi2 = self.critic(si, ai)
        
        if self.Qb_strain_critic:
            if self.Qb_strain_max:
                Lstrain = self.lamb1.detach() * (back_up_b1 - back_up_d).mean() + self.lamb2.detach() * (back_up_b2 - back_up_d).mean() + \
                            self.lamb3.detach() * (back_up_d_max - back_up).mean()
            else:
                Lstrain = self.lamb1.detach() * (back_up_b1 - back_up_d).mean() + self.lamb2.detach() * (back_up_b2 - back_up_d).mean()
            #Lc = F.mse_loss(qi1, yi) + F.mse_loss(qi2, yi)
            Lc = ((qi1 - yi) ** 2).mean() + ((qi2 - yi) ** 2).mean()
            Lall = Lc + Lstrain
            # print("Lstrain:", Lstrain.item())
            # print("Lc:", Lc)
        else:
            #Lc = F.mse_loss(qi1, yi) + F.mse_loss(qi2, yi)
            Lc = ((qi1 - yi) ** 2).mean() + ((qi2 - yi) ** 2).mean()
            Lall = Lc

        self.optimizer_critic.zero_grad()
        Lall.backward()
        self.optimizer_critic.step()

        if self.Qb_strain_critic:
            # print('lamb:', self.lamb1.item(), self.lamb2.item())
            self.log_lamb_optimizer.zero_grad()
            if self.Qb_strain_max:
                lamb_loss = (self.lamb1 * (back_up_d - back_up_b1).detach()).mean() + (self.lamb2 * (back_up_d - back_up_b2).detach()).mean() + \
                            (self.lamb3 * (back_up - back_up_d_max).detach()).mean()
            else:
                lamb_loss = (self.lamb1 * (back_up_d - back_up_b1).detach()).mean() + (self.lamb2 * (back_up_d - back_up_b2).detach()).mean()
            lamb_loss.backward()
            self.log_lamb_optimizer.step()
        
        return Lc

    def update_actor_and_alpha(self, oi, si, si_p = None, base_action=None, update_alpha=True):
        
        if base_action is not None:
            with torch.no_grad():
                q_ai_d1, q_ai_d2 = self.critic(si, base_action)
                q_ai_d = torch.min(q_ai_d1, q_ai_d2)

            if self.use_fast:
                mu, a, log_pi, log_std = self.actor(si)
            else:
                mu, a, log_pi, log_std = self.actor(oi, si_p)
            q_a1, q_a2 = self.critic(si, a)
            q_a = torch.min(q_a1, q_a2)

            with torch.no_grad():
                if self.Qb_strain_actor and np.random.uniform(0, 1) < self.epsilon:
                    xi = torch.ones_like(q_a)
                else:
                    xi = nn.ReLU()(torch.sign(q_ai_d - q_a))
            
            if self.use_mu:
                Lbc = (((mu - base_action) ** 2).mean(dim=1, keepdim=True) * xi).sum() / max(xi.sum().item(), 1)
            else:
                Lbc = (((a - base_action) ** 2).mean(dim=1, keepdim=True) * xi).sum() / max(xi.sum().item(), 1)

            actor_loss = (self.alpha.detach() * log_pi - q_a).mean()
            La = Lbc + 0.02 * actor_loss 

            self.optimizer_actor.zero_grad()
            La.backward()
            self.optimizer_actor.step()
        else:
            if self.use_fast:
                mu, a, log_pi, log_std = self.actor(si)
            else:
                mu, a, log_pi, log_std = self.actor(oi, si_p)
            q_a1, q_a2 = self.critic(si, a)
            q_a = torch.min(q_a1, q_a2)
            La = 0.02 * (self.alpha.detach() * log_pi - q_a).mean()
            Lbc = 0

            self.optimizer_actor.zero_grad()
            La.backward()
            self.optimizer_actor.step()

        if update_alpha:
            # print('entropy:', -log_pi)
            # print("target:", self.target_entropy)
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        return La, Lbc

    def train(self, frame):
        #TODO 保留吗？
        # if step % self.update_freq != 0:
        #     return
        total_Lc = total_La = total_Lbc = 0
        steps = min(int(frame), max((5 * self.buffer.size) // self.batch_size, 1))
        for i in range(steps):
            batch = self.buffer.sample_batch(batch_size=self.batch_size)
            si = np_to_tensor(batch['sta1'], self.device)
            sn = np_to_tensor(batch['sta2'], self.device)
            ai = np_to_tensor(batch['acts'], self.device)
            ri = np_to_tensor(batch['rews'], self.device)
            d = np_to_tensor(batch['done'], self.device)
            
            if not self.use_fast:
                oi = np_to_tensor(batch['obv1'], self.device)
                on = np_to_tensor(batch['obv2'], self.device)
                si_p = np_to_tensor(batch['sta1'][:, :8], self.device)
                sn_p = np_to_tensor(batch['sta2'][:, :8], self.device)

                if self.color_jitter and frame > 0:#self.frame_num:
                    oi = random_color_jitter(oi)
            
            base_action = np_to_tensor(base_controller(batch['sta1'], self.base), self.device)
            base_action_n = np_to_tensor(base_controller(batch['sta2'], self.base), self.device)

            if self.behavior_clone:
                if self.use_fast:
                    Lc = self.update_critic(oi, si, ai, ri, on, sn, d, base_action_n)
                    La, Lbc = self.update_actor_and_alpha(oi.detach(), si, base_action)
                else:
                    Lc = self.update_critic(oi, si, ai, ri, on, sn, d, base_action_n, sn_p)#, base_action, si_p)
                    La, Lbc = self.update_actor_and_alpha(oi.detach(), si, si_p, base_action)
            else:
                if self.use_fast:
                    Lc = self.update_critic(oi, si, ai, ri, on, sn, d, None)
                    La, _ = self.update_actor_and_alpha(oi.detach(), si, None)
                else:
                    Lc = self.update_critic(oi, si, ai, ri, on, sn, d, None, sn_p)
                    La, _ = self.update_actor_and_alpha(oi.detach(), si, si_p, None)
            
            total_Lc += Lc.item()
            if self.behavior_clone:
                total_Lbc += Lbc.item()
            total_La += La.item()

            ### 只有critic用targetupdate
            soft_update(self.target_critic, self.critic, self.tau)

        return total_Lc / steps, total_La / steps, total_Lbc / steps