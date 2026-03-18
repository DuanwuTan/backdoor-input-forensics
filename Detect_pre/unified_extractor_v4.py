import sys
import os
import torch
import torch.nn as nn
import numpy as np
import collections
from tqdm import tqdm
from torch.utils.data import DataLoader

# 1. 强力路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. 导入逻辑 (完全套用 V3 的成功模板)
from utils.save_load_attack import load_attack_result
try:
    from models.resnet import resnet18
except ImportError:
    from torchvision.models import resnet18

# 3. V4 核心特征提取函数 (质心特征)
def get_wanet_v4_features(o):
    """
    针对 WaNet 的专项特征：空间质心 + 峰均比
    输入 o: (B, C, H, W)
    """
    B, C, H, W = o.shape
    device = o.device
    
    # 准备坐标网格
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid_y = grid_y.to(device).float() / H
    grid_x = grid_x.to(device).float() / W
    
    # 防止除以 0
    eps = 1e-6
    channel_sum = torch.sum(o, dim=(2, 3), keepdim=True) + eps
    
    # 计算空间质心
    centroid_y = torch.sum(o * grid_y, dim=(2, 3)) / channel_sum.view(B, C)
    centroid_x = torch.sum(o * grid_x, dim=(2, 3)) / channel_sum.view(B, C)
    
    # 计算峰均比 (PAPR)
    ch_max, _ = torch.max(o.view(B, C, -1), dim=2)
    ch_mean = torch.mean(o, dim=(2, 3))
    papr = ch_max / (ch_mean + eps)
    
    # 返回拼接后的特征 (B, C * 3)
    return torch.cat([centroid_y, centroid_x, papr], dim=1)

# 4. 提取执行逻辑
def extract_v4(attack_name, result_path, device='cuda'):
    print(f"\n>>> 正在提取 V4 特征: {attack_name}")
    result = load_attack_result(result_path)
    
    # 实例化并加载模型
    model_data = result['model']
    if isinstance(model_data, (dict, collections.OrderedDict)):
        print("检测到状态字典，正在实例化 resnet18...")
        # 针对 CIFAR10 默认 10 分类
        model = resnet18(num_classes=10)
        model.load_state_dict(model_data)
    else:
        model = model_data
    
    model = model.to(device).eval()

    # 数据加载器 (使用 V3 的兼容写法)
    clean_loader = DataLoader(result['clean_test'], batch_size=64, shuffle=False)
    bd_loader = DataLoader(result['bd_test'], batch_size=64, shuffle=False)
    
    features = {f'layer{i}': [] for i in range(1, 5)}
    handles = []

    def get_hook(name):
        def hook(m, i, o):
            features[name].append(get_wanet_v4_features(o).cpu().numpy())
        return hook

    for l in features:
        handles.append(getattr(model, l).register_forward_hook(get_hook(l)))

    print("提取干净样本...")
    with torch.no_grad():
        for batch in tqdm(clean_loader):
            model(batch[0].to(device))
            
    save_dir = f'./features_v4/{attack_name}'
    os.makedirs(save_dir, exist_ok=True)
    for l in features:
        np.save(f'{save_dir}/{l}_clean.npy', np.concatenate(features[l], axis=0))
        features[l] = [] 

    print("提取后门样本...")
    with torch.no_grad():
        for batch in tqdm(bd_loader):
            model(batch[0].to(device))
            
    for l in features:
        np.save(f'{save_dir}/{l}_bd.npy', np.concatenate(features[l], axis=0))

    for h in handles: h.remove()
    print(f"提取完成！保存在 {save_dir}")

if __name__ == '__main__':
    wanet_path = './record/20260221_013811_wanet_attack_wanet_CF3H/attack_result.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.path.exists(wanet_path):
        extract_v4('wanet', wanet_path, device=device)
    else:
        print(f"路径不存在: {wanet_path}")