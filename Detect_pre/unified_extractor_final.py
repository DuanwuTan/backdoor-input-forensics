import os
import sys
import torch
import torch.nn as nn
import numpy as np
import collections
from tqdm import tqdm
from torch.utils.data import DataLoader

# =========================================================
# 1. 强力路径修复 (确保能找到 models 和 utils)
# =========================================================
current_file = os.path.abspath(__file__)
detect_pre_dir = os.path.dirname(current_file)
project_root = os.path.dirname(detect_pre_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入本地模块
from utils.save_load_attack import load_attack_result
try:
    from models.resnet import resnet18
except ImportError:
    from torchvision.models import resnet18

# =========================================================
# 2. UCAT 终极特征融合逻辑 (Mean + Gini + Centroid + PAPR)
# =========================================================
def get_final_features(o):
    """
    UCAT 核心特征集：
    - Mean (V1): 捕获强度异常 (SIG, BadNets)
    - Gini (V2): 捕获激活极化 (InputAware)
    - Centroid (V4): 捕获位置偏移 (WaNet)
    - PAPR: 捕获响应尖锐度变化 (Warping/Refool)
    """
    B, C, H, W = o.shape
    device = o.device
    eps = 1e-6

    # --- 1. 强度与分布特征 ---
    mu = torch.mean(o, dim=(2, 3))
    std = torch.std(o, dim=(2, 3))
    gini = std / (mu + eps)

    # --- 2. 拓扑质心特征 ---
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid_y = grid_y.to(device).float() / H
    grid_x = grid_x.to(device).float() / W
    
    channel_sum = torch.sum(o, dim=(2, 3), keepdim=True) + eps
    centroid_y = torch.sum(o * grid_y, dim=(2, 3)) / channel_sum.view(B, C)
    centroid_x = torch.sum(o * grid_x, dim=(2, 3)) / channel_sum.view(B, C)

    # --- 3. 峰值响应特征 ---
    ch_max, _ = torch.max(o.view(B, C, -1), dim=2)
    papr = ch_max / (mu + eps)

    # 拼接全能特征: (B, C * 5)
    return torch.cat([mu, gini, centroid_y, centroid_x, papr], dim=1)

# =========================================================
# 3. 提取执行函数
# =========================================================
def extract_final(attack_name, result_path, device='cuda'):
    print(f"\n>>> 正在提取 UCAT 终极特征: {attack_name}")
    result = load_attack_result(result_path)
    
    # 模型处理
    model_data = result['model']
    if isinstance(model_data, (dict, collections.OrderedDict)):
        print(f"[{attack_name}] 检测到状态字典，正在实例化 resnet18...")
        model = resnet18(num_classes=10)
        model.load_state_dict(model_data)
    else:
        model = model_data
    model = model.to(device).eval()

    # 数据加载器
    loaders = {
        'clean': DataLoader(result['clean_test'], batch_size=64, shuffle=False),
        'bd': DataLoader(result['bd_test'], batch_size=64, shuffle=False)
    }
    
    features = {f'layer{i}': {'clean': [], 'bd': []} for i in range(1, 5)}
    
    def get_hook(name, mode):
        def hook(m, i, o):
            features[name][mode].append(get_final_features(o).cpu().numpy())
        return hook

    save_dir = f'./features_final/{attack_name}'
    os.makedirs(save_dir, exist_ok=True)

    # 分别提取干净和后门样本
    for mode in ['clean', 'bd']:
        print(f"正在处理 {attack_name} 的 {mode} 样本...")
        handles = [getattr(model, l).register_forward_hook(get_hook(l, mode)) for l in features]
        with torch.no_grad():
            for batch in tqdm(loaders[mode]):
                img = batch[0] # 兼容各种返回长度
                model(img.to(device))
        for h in handles:
            h.remove()

    # 批量保存
    for l in features:
        np.save(f'{save_dir}/{l}_clean.npy', np.concatenate(features[l]['clean'], axis=0))
        np.save(f'{save_dir}/{l}_bd.npy', np.concatenate(features[l]['bd'], axis=0))
    
    print(f"✅ {attack_name} 提取完成！")

# =========================================================
# 4. 主程序入口
# =========================================================
if __name__ == '__main__':
    attacks = {
        'badnets': './record/20260215_235930_badnet_attack_badnet_DvMv/attack_result.pt',
        'blended': './record/20260221_010831_blended_attack_blended_Aniv/attack_result.pt',
        'wanet': './record/20260221_013811_wanet_attack_wanet_CF3H/attack_result.pt',
        'sig': './record/sig_attack_1/attack_result.pt',
        'refool': './record/refool_attack_2/attack_result.pt',
        'inputaware': './record/inputaware_attack_1/attack_result.pt'
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    for name, path in attacks.items():
        if os.path.exists(path):
            try:
                extract_final(name, path, device=device)
            except Exception as e:
                print(f"❌ 提取 {name} 失败: {e}")
        else:
            print(f"跳过 {name}，未找到模型文件。")

    print("\n🎉 所有攻击的终极特征已保存在 ./features_final/ 文件夹下！")