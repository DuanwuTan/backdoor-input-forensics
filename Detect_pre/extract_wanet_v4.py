import torch
import numpy as np

def get_wanet_v4_features(o):
    """
    针对 WaNet 的专项特征：空间质心 + 峰均比
    输入 o: (B, C, H, W)
    """
    B, C, H, W = o.shape
    device = o.device
    
    # 1. 准备坐标网格
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid_y = grid_y.to(device).float() / H
    grid_x = grid_x.to(device).float() / W
    
    # 为了防止全 0 通道除以 0，加一个 epsilon
    eps = 1e-6
    channel_sum = torch.sum(o, dim=(2, 3), keepdim=True) + eps
    
    # 2. 计算空间质心 (Center of Mass)
    # 每个通道的激活重心在哪里？
    centroid_y = torch.sum(o * grid_y, dim=(2, 3)) / channel_sum.squeeze()
    centroid_x = torch.sum(o * grid_x, dim=(2, 3)) / channel_sum.squeeze()
    
    # 3. 计算峰均比 (PAPR)
    # 几何形变会改变响应的“尖锐度”
    ch_max, _ = torch.max(o.view(B, C, -1), dim=2)
    ch_mean = torch.mean(o, dim=(2, 3))
    papr = ch_max / (ch_mean + eps)
    
    # 合并特征 (B, C * 3)
    return torch.cat([centroid_y, centroid_x, papr], dim=1)
