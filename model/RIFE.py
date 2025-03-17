import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW, Adam
import torch.optim as optim
import torch.nn.functional as F
from model.IFNet import *
from model.loss import LapLoss
from common import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Model:
    def __init__(self):
        self.flownet = IFNet()
        self.device()
        # self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3) # use large weight decay may avoid NaN loss
        self.optimG = AdamW(self.flownet.parameters(), lr=1e-6, weight_decay=1e-3)
        self.lap = LapLoss() # 拉普拉斯损失 

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path): # 这个还得是斟酌一下
        self.flownet.load_state_dict(torch.load('{}//flownet.pkl'.format(path)))
        
    def save_model(self, path):
        torch.save(self.flownet.state_dict(),'{}//flownet_new.pkl'.format(path))

    def inference(self, img0, img1, scale=1, scale_list=[4, 2, 1], TTA=False):
        for i in range(3): # scale 可以用来调整分辨率, 让原图尽可能还是靠近训练数据吧
            scale_list[i] = scale_list[i] * 1.0 / scale
        imgs = torch.cat((img0, img1), 1)
        flow, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(imgs, scale_list)
        if TTA == False: # 测试时候增强, 翻转着再来一次
            return merged[2]
        else:
            flow2, merged2, flow_teacher2, merged_teacher2, loss_distill2 = self.flownet(imgs.flip(2).flip(3), scale_list)
            return (merged[2] + merged2[2].flip(2).flip(3)) / 2
    
    def update(self, imgs, gt, learning_rate=0, training=True):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate # 所以初始化的 optimG.param_groups 的 lr 没有什么意义 
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        flow, merged, flow_teacher, merged_teacher, loss_distill = self.flownet(torch.cat((imgs, gt), 1), scale=[4, 2, 1])
        loss_l1 = (self.lap(merged[2], gt)).mean()
        loss_tea = (self.lap(merged_teacher, gt)).mean()
        
        if training:
            self.optimG.zero_grad()
            loss_G = loss_l1 + loss_tea + loss_distill * 0.01 
            loss_G.backward()
            self.optimG.step()
            
        return merged[2], {
            'merged_tea': merged_teacher,
            'flow': flow[2][:, :2],
            'flow_tea': flow_teacher[:, :2],
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_distill': loss_distill,
            }
