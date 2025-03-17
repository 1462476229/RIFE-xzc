import os
import cv2
import math
import torch
import numpy as np
import random
import argparse

from model.RIFE import Model
from dataset import VimeoDataset
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda")


def get_learning_rate(step): # 学习率, warmup 暂时没有实现
    return 3e-6
    warmup_step = 2000.
    if step < warmup_step:
        mul = step / warmup_step
        return 3e-4 * mul
    else:
        mul = np.cos((step - warmup_step) / (args.epoch * args.epoch_per_step - warmup_step) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6

def flow2rgb(flow_map_np): # 把光流图可视化, 需要一个归一化
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def train(model):
    step = 0
    dataset = VimeoDataset('train')
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=1)
    args.epoch_per_step = train_data.__len__()
    print(f"args.epoch_per_step : {train_data.__len__()}")
    dataset_val = VimeoDataset('validation') # validation 只占了 0.05 的数据
    val_data = DataLoader(dataset_val, batch_size=16, num_workers=1)
    print('training...')
    
    for epoch in range(args.epoch):
        for i, data in enumerate(train_data):

            data_gpu = data
            data_gpu = data_gpu.to(device) / 255. # 归一化到了 [0,1]
            imgs = data_gpu[:, :6] # (I0, I1)
            gt = data_gpu[:, 6:9] # (GT)
            learning_rate = get_learning_rate(step)
            pred, info = model.update(imgs, gt, learning_rate, training=True) # 训练的核心代码
            if i % 100 == 0:
                print('epoch:{} {}/{} loss_l1:{:.4f} loss_tea:{:.4f} loss_distill:{:.4f}'.format
                      (epoch, i, len(train_data), float(info['loss_l1']), float(info['loss_tea']), float(info['loss_distill'])))
            step += 1
        if epoch % 5 == 0:
            evaluate(model, val_data, step)
            
        model.save_model(args.save_model_dir)  

def evaluate(model, val_data, step):
    save_dir = args.save_dir
    loss_l1_list = []
    loss_distill_list = []
    loss_tea_list = []
    psnr_list = [] # Peak Signal-to-Noise Ratio
    psnr_list_teacher = []
    
    for i, data in enumerate(val_data):
        data_gpu = data
        data_gpu = data_gpu.to(device) / 255.        
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]
        with torch.no_grad():
            pred, info = model.update(imgs, gt, training=False)
            merged_img = info['merged_tea']
        loss_l1_list.append(info['loss_l1'].cpu().numpy())
        loss_tea_list.append(info['loss_tea'].cpu().numpy())
        loss_distill_list.append(info['loss_distill'].cpu().numpy())
        for j in range(gt.shape[0]):
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
            psnr = -10 * math.log10(torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
            psnr_list_teacher.append(psnr)
        gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        merged_img = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy() # 这里只返回了 2 通道, 只有一个方向
        flow1 = info['flow_tea'].permute(0, 2, 3, 1).cpu().numpy() # 修改后也是 2 通道
        if i == 0:
            for j in range(10):
                # Concatenate images horizontally (GT, Prediction, Merged Image)
                img_concat = np.concatenate((merged_img[j], pred[j], gt[j]), axis=1)  # BGR for saving
                img_path = os.path.join(save_dir, f'img_{j}.png')
                cv2.imwrite(img_path, img_concat)
                # Convert and save optical flow using OpenCV
                # print(f"info['flow'].shape: {info['flow'].shape}, info['flow_tea'].shape: {info['flow_tea'].shape}")
                flow0_rgb = (flow2rgb(flow0[j]) * 255).astype('uint8')
                flow1_rgb = (flow2rgb(flow1[j]) * 255).astype('uint8')
                flow_concat = np.concatenate((flow0_rgb, flow1_rgb), axis=1)
                flow_path = os.path.join(save_dir, f'flow_{j}.png')
                cv2.imwrite(flow_path, flow_concat) # BGR 3通道
        break

    psnr_avg = np.array(psnr_list).mean()
    psnr_teacher_avg = np.array(psnr_list_teacher).mean()
    with open(os.path.join(save_dir, f'psnr.txt'), 'w') as f:
        f.write(f'PSNR (student): {psnr_avg}\n')
        f.write(f'PSNR (teacher): {psnr_teacher_avg}\n')
    
    
if __name__ == "__main__":    
    print(f"当前cuda设备是{torch.cuda.current_device()} 是否可以用 {torch.cuda.is_available()}", )
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=32, type=int, help='minibatch size')
    parser.add_argument('--save_dir', default="C:\\Users\\Administrator\\Desktop\\xzc-RIFE\\output", type=str, help='save output dir')
    parser.add_argument('--save_model_dir', default='C:\\Users\\Administrator\\Desktop\\xzc-RIFE\\train_log', type=str, help='save model dir')
    parser.add_argument('--model_dir', default='C:\\Users\\Administrator\\Desktop\\xzc-RIFE\\train_log', type=str, help='load model dir')    
    args = parser.parse_args()
    seed = 20250312
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True # 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法
    model = Model()
    if args.model_dir != None:
        model.load_model(args.model_dir)
    train(model)
        
