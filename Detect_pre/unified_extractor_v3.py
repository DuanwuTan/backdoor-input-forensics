import sys
import os

# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取 Detect_pre 所在的目录
detect_pre_dir = os.path.dirname(current_script_path)
# 获取项目根目录 (BackdoorBench)
project_root = os.path.dirname(detect_pre_dir)

# 将项目根目录加入 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.save_load_attack import load_attack_result
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

# 假设你已经有了 BackdoorBench 的相关引用
# 如果报错，请确保路径正确
from utils.save_load_attack import load_attack_result
import collections
# 尝试导入 BackdoorBench 的 ResNet 定义
try:
    from models.resnet import resnet18
except ImportError:
    # 如果路径不对，尝试从 torchvision 导入（CIFAR10 版本的 ResNet18 结构通常一致）
    from torchvision.models import resnet18
def get_v3_features(feature_map):
    """
    输入: (B, C, H, W)
    输出: (B, C * 4) 包含: Mean, Gini, H-Grad, V-Grad
    """
    B, C, H, W = feature_map.shape
    
    # 1. Channel Mean
    mu = torch.mean(feature_map, dim=(2, 3)) # (B, C)
    
    # 2. Gini Index (Approx by CV)
    std = torch.std(feature_map, dim=(2, 3))
    gini = std / (mu + 1e-6) # (B, C)
    
    # 3. Spatial Gradients (攻克 WaNet 的核心)
    # 水平梯度：像素间的差异
    grad_h = torch.abs(feature_map[:, :, :, 1:] - feature_map[:, :, :, :-1])
    grad_h_mu = torch.mean(grad_h, dim=(2, 3)) # (B, C)
    
    # 垂直梯度
    grad_v = torch.abs(feature_map[:, :, 1:, :] - feature_map[:, :, :-1, :])
    grad_v_mu = torch.mean(grad_v, dim=(2, 3)) # (B, C)
    
    # 合并特征
    return torch.cat([mu, gini, grad_h_mu, grad_v_mu], dim=1)

def extract_all_v3(attack_name, result_path, device='cuda'):
    print(f"\n>>> 正在提取 V3 特征: {attack_name}")
    result = load_attack_result(result_path)
    
    # --- 修复逻辑开始 ---
    model_data = result['model']
    
    if isinstance(model_data, (dict, collections.OrderedDict)):
        # 如果是状态字典，我们需要实例化模型再加载
        print(f"检测到状态字典，正在实例化 resnet18...")
        # 默认使用 CIFAR10 的 10 分类，如果有 args 可以动态获取
        num_classes = 10
        if 'args' in result and hasattr(result['args'], 'num_classes'):
            num_classes = result['args'].num_classes
            
        model = resnet18(num_classes=num_classes)
        model.load_state_dict(model_data)
    else:
        # 如果已经是模型对象
        model = model_data
    # --- 修复逻辑结束 ---

    model = model.to(device).eval()
    clean_loader = DataLoader(result['clean_test'], batch_size=64, shuffle=False)
    bd_loader = DataLoader(result['bd_test'], batch_size=64, shuffle=False)
    
    # 定义 Hook 提取 ResNet 四个层
    features = {layer: [] for layer in ['layer1', 'layer2', 'layer3', 'layer4']}
    handles = []
    
    def get_hook(name):
        def hook(m, i, o):
            features[name].append(get_v3_features(o).cpu().numpy())
        return hook

    handles.append(model.layer1.register_forward_hook(get_hook('layer1')))
    handles.append(model.layer2.register_forward_hook(get_hook('layer2')))
    handles.append(model.layer3.register_forward_hook(get_hook('layer3')))
    handles.append(model.layer4.register_forward_hook(get_hook('layer4')))

    # 提取干净样本
    print("提取干净样本...")
    with torch.no_grad():
        for batch in tqdm(clean_loader):
            img = batch[0] # 第一个永远是图像
            model(img.to(device))
            
    for layer in features:
        save_dir = f'./features_v3/{attack_name}'
        os.makedirs(save_dir, exist_ok=True)
        np.save(f'{save_dir}/{layer}_clean.npy', np.concatenate(features[layer], axis=0))
        features[layer] = [] # 清空缓存

    # 提取后门样本
    print("提取后门样本...")
    with torch.no_grad():
        for batch in tqdm(bd_loader):
            img = batch[0] # 第一个永远是图像
            model(img.to(device))
    for layer in features:
        save_dir = f'./features_v3/{attack_name}'
        np.save(f'{save_dir}/{layer}_bd.npy', np.concatenate(features[layer], axis=0))
        features[layer] = []

    for h in handles: h.remove()
if __name__ == '__main__':
    # 这里的路径完全对应你交接文档中的模型位置
    attacks = {
        'badnets': './record/20260215_235930_badnet_attack_badnet_DvMv/attack_result.pt',
        'blended': './record/20260221_010831_blended_attack_blended_Aniv/attack_result.pt',
        'wanet': './record/20260221_013811_wanet_attack_wanet_CF3H/attack_result.pt',
        'sig': './record/sig_attack_1/attack_result.pt',
        'refool': './record/refool_attack_2/attack_result.pt',
        'inputaware': './record/inputaware_attack_1/attack_result.pt'
    }

    # 检查 CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 循环提取
    for name, path in attacks.items():
        if os.path.exists(path):
            try:
                extract_all_v3(name, path, device=device)
            except Exception as e:
                print(f"提取 {name} 时出错: {e}")
        else:
            print(f"跳过 {name}，路径不存在: {path}")

    print("\n所有 V3 特征提取完成！保存于 ./features_v3/")
# 主循环执行 6 种攻击的提取...