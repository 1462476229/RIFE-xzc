import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gauss_kernel(size=5, channels=3):
    # 暂时只支持 3, 5 两个大小的高斯核
    if size == 5:
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                                [4., 16., 24., 16., 4.],
                                [6., 24., 36., 24., 6.],
                                [4., 16., 24., 16., 4.],
                                [1., 4., 6., 4., 1.]])
        kernel /= 256. # shape: [5,5]
    elif size == 3:
        kernel = torch.tensor([[1., 2., 1],
                                [2., 4., 2.],
                                [1., 2., 1.]])
        kernel /= 16.
    else:
        raise NotImplementedError
    
    kernel = kernel.repeat(channels, 1, 1, 1) # shape: [3,1,5,5]
    kernel = kernel.to(device)
    return kernel

def downsample(x): # 下采样层, 直接除以 2 的大小
    return x[:, :, ::2, ::2]

def upsample(x): # 上采样, 中间填 0, 然后 4 倍数的高斯核滤波 
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1]))

def conv_gauss(img, kernel): # 高斯滤波
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5, channels=3): # 下采样 5 层
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels)
        
    def forward(self, input, target):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))


if __name__ == "__main__":
    pass