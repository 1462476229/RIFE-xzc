import torch
import torch.nn as nn
import torch.nn.functional as F

from common import *
from model.refine import *

class IFBlock(nn.Module):
    def __init__(self, input_channel, c=64) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(input_channel, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1) # 输出 5 通道
    def forward(self, x, flow, scale):
        # if flow != None:
        #     print(f"x shape是 {x.shape}  flow shape是 {flow.shape}  scale是 {scale}")
        # scale 就是那个分辨率参数, scale 越大, 图片分辨率越小
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
            # x 先做下采样
        if flow != None:
            if scale != 1:
                # 光流图在下采样时候, 也得将相对距离缩放相同比例
                flow = F.interpolate(flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1./scale #flow !!!
            # print(f"x shape是 {x.shape}, flow shape是 {flow.shape}")
            x = torch.cat((x, flow), dim=1) # 在第二维 concat 在一起
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x) # 还没能回到原大小
        tmp = F.interpolate(tmp, scale_factor = scale * 2, mode="bilinear", align_corners=False) # 回到原本大小
        flow = tmp[:, :4] * scale * 2 # 需要两个flow, 一个表示x轴, 一个表示y轴, 才能完美表示光流
        mask = tmp[:, 4:5] # mask 矩阵
        # print(f"x shape是 {x.shape}, flow shape是 {flow.shape}")
        return flow, mask

class IFNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block0 = IFBlock(6, c=240) #(I0, I1)
        self.block1 = IFBlock(13 + 4, c=150) #(I0, I1, I0->t,I1->t, mask, flow1, flow2)
        self.block2 = IFBlock(13 + 4, c=90) #(I0, I1, I0->t,I1->t, mask, flow1, flow2)
        self.block_tea = IFBlock(16 + 4, c=90) #(I0, I1, I0->t,I1->t, mask, gt, flow1, flow2)
        self.contextnet = Contextnet()
        self.unet = Unet()

    def forward(self, x, scale=[4,2,1]):
        I0 = x[:, :3] # 得到原始输入图片
        I1 = x[:, 3:6] 
        gt = x[:, 6:] # 如果有之后的, 则是 gt
        # print(f"Gt.shape : {gt.shape}")
        flow_list = []
        merged = []
        mask_list = []
        warped_I0 = I0
        warped_I1 = I1
        flow = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            # print(f"test:  I0.shape {I0.shape}  I1.shape {I1.shape} warped_I0.shape {warped_I0.shape} warped_I1.shape {warped_I1.shape} ")
            if flow != None:
                flow_d, mask_d = stu[i](torch.cat((I0, I1, warped_I0, warped_I1, mask), 1), flow, scale=scale[i])
                flow = flow + flow_d # 修正之后的 flow
                mask = mask + mask_d # 修正之后的 mask
            else:
                flow, mask = stu[i](torch.cat((I0, I1), 1), None, scale=scale[i])
            
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            # print(f"test:  flow.shape {flow.shape}")
            warped_I0 = warp(I0, flow[:, :2]) # 根据光流得到变化后的图像
            warped_I1 = warp(I1, flow[:, 2:4]) # 根据光流得到变化后的图像
            merged.append((warped_I0, warped_I1)) # 将每次变化的也存下来
        if gt.shape[1] == 3: # 如果提供了GT, 则说明是在有蒸馏的训练, GT 也只能暴露给 teacher 网络
            flow_d, mask_d = self.block_tea(torch.cat((I0, I1, warped_I0, warped_I1, mask, gt), 1), flow, scale=1)
            flow_teacher = flow + flow_d
            warped_I0_teacher = warp(I0, flow_teacher[:, :2])
            warped_I1_teacher = warp(I1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask + mask_d)
            merged_teacher = warped_I0_teacher * mask_teacher + warped_I1_teacher * (1 - mask_teacher) # 得到了合并后的图像
        else:
            flow_teacher = None
            merged_teacher = None
            
        for i in range(3):
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i]) # 得到了合并后的图像
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True) + 0.01).float().detach()
                loss_distill += (((flow_teacher.detach() - flow_list[i]) ** 2).mean(1, True) ** 0.5 * loss_mask).mean()
        
        c0 = self.contextnet(I0, flow[:, :2])
        c1 = self.contextnet(I1, flow[:, 2:4])
        tmp = self.unet(I0, I1, warped_I0, warped_I1, mask, flow, c0, c1)
        res = tmp[:, :3] * 2 - 1 # 因为 unet 的返回是 sigmoid(x) , 这里将其扩大到 [-1,1]
        merged[2] = torch.clamp(merged[2] + res, 0, 1) # 钳制到 0~1 之间
        return flow_list, merged, flow_teacher, merged_teacher, loss_distill

 
