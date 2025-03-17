import os
import cv2
import ast
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class VimeoDataset(Dataset):
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name  # train, test, validation
        self.h = 256 # 所需训练的图片的 H，感觉这里的定义没什么用，毕竟后面在拆分的时候变成了 [244, 244]
        self.w = 448 # 所需训练的图片的 W
        self.data_root = 'E:\\NOW\\vimeo_triplet' # 三元组
        self.image_root = os.path.join(self.data_root, 'sequences')
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()# 按行拆分
        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        # 这个时候的 dataset 什么的都是 List[Str] 类型的暂时
        cnt = int(len(self.trainlist) * 0.95)
        if self.dataset_name == 'train':
            self.meta_data = self.trainlist[:cnt] # train 所需的数据集
        elif self.dataset_name == 'test':
            self.meta_data = self.testlist # test 所需的数据集是全量数据集
        else:
            self.meta_data = self.trainlist[cnt:] # 这个可以称作 validation
           
    def crop(self, img0, gt, img1, h, w): # 此时还是整数像素
        ih, iw, _ = img0.shape # 裁剪出一个 h,w 大小的图片
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        return img0, gt, img1 # 

    def getimg(self, index):
        imgpath = os.path.join(self.image_root, self.meta_data[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']

        # Load images
        img0 = cv2.imread(imgpaths[0]) #[h, w, c]
        gt = cv2.imread(imgpaths[1]) #[h, w, c]
        img1 = cv2.imread(imgpaths[2]) #[h, w, c]
        return img0, gt, img1

            
    def __getitem__(self, index):        
        img0, gt, img1 = self.getimg(index) # 读取成图片 
        # 有个问题, timestep 就默认为 0.5 了
        if self.dataset_name == 'train': # 只是 train 才会使用增强
            img0, gt, img1 = self.crop(img0, gt, img1, 224, 224)
            if random.uniform(0, 1) < 0.5: # 反转颜色
                img0 = img0[:, :, ::-1]
                img1 = img1[:, :, ::-1]
                gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5: # 反转上下
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
            if random.uniform(0, 1) < 0.5: # 反转左右
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
            if random.uniform(0, 1) < 0.5: # 反正动作顺序(虽然有可能不真实, 但是补帧其实不在意)
                tmp = img1
                img1 = img0
                img0 = tmp
            p = random.uniform(0, 1) # 旋转
            if p < 0.25:  
                img0 = cv2.rotate(img0, cv2.ROTATE_90_CLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_CLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            elif p < 0.5:
                img0 = cv2.rotate(img0, cv2.ROTATE_180)
                gt = cv2.rotate(gt, cv2.ROTATE_180)
                img1 = cv2.rotate(img1, cv2.ROTATE_180)
            elif p < 0.75:
                img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
                gt = cv2.rotate(gt, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1) # 顺序改为 [c, h, w]
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)
        # concat 之后相当于 [c * 3, h , w]
        return torch.cat((img0, img1, gt), 0)
