import os
import sys
import torch
import torch.nn as nn
import numpy as np
import collections
from tqdm import tqdm
from torch.utils.data import DataLoader

# =========================================================
# 1. 强力路径修复
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
    B, C, H, W = o.shape
    device = o.device
    eps = 1e-6

    # --- 1. 强度与分布特征 ---
    mu = torch.mean(o, dim=(2, 3))
    # 针对 Layer 4 (1x1) 的 std 保护
    if H > 1:
        std = torch.std(o, dim=(2, 3))
    else:
        std = torch.zeros_like(mu)
    gini = std / (mu + eps)

    # --- 2. 拓扑质心特征 ---
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid_y = grid_y.float() / H
    grid_x = grid_x.float() / W
    
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
    
    # --- 架构适配逻辑 ---
    model_data = result['model']
    if attack_name == 'bpp':
        print(f"[{attack_name}] 检测到 PreActResNet18 架构需求，正在切换...")
        from models.preact_resnet import PreActResNet18
        model = PreActResNet18(num_classes=10)
    else:
        print(f"[{attack_name}] 使用标准 ResNet18...")
        model = resnet18(num_classes=10)

    # 统一加载权重 (处理 module. 前缀)
    if isinstance(model_data, (dict, collections.OrderedDict)):
        new_sd = collections.OrderedDict()
        for k, v in model_data.items():
            nk = k[7:] if k.startswith('module.') else k
            nk = nk.replace('linear.', 'fc.') # 适配不同命名空间
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)
    else:
        model = model_data
    
    model = model.to(device).eval()

    # --- 数据集搜索逻辑 ---
    # 尝试从 result 中获取后门数据集，涵盖 LIRA 可能的命名方式
    bd_dataset = result.get('bd_test') or result.get('test_bd')
    if bd_dataset is None:
        print(f"⚠️ 警告: {attack_name} 未在根目录找到 bd_test，尝试扫描 keys: {result.keys()}")
    
    loaders = {
        'clean': DataLoader(result['clean_test'], batch_size=64, shuffle=False),
        'bd': DataLoader(bd_dataset, batch_size=64, shuffle=False) if bd_dataset else None
    }
    
    if loaders['bd'] is None:
        print(f"❌ 严重错误: {attack_name} 无法找到后门数据集，AUROC 将为 0.5！")
        return

    features = {f'layer{i}': {'clean': [], 'bd': []} for i in range(1, 5)}
    
    def get_hook(name, mode):
        def hook(m, i, o):
            features[name][mode].append(get_final_features(o).cpu().numpy())
        return hook

    save_dir = f'./features_final/{attack_name}'
    os.makedirs(save_dir, exist_ok=True)
    
    # 顺序提取
    for mode in ['clean', 'bd']:
        print(f"正在收割 {attack_name} 的 {mode} 样本特征...")
        # 注册钩子
        handles = [getattr(model, l).register_forward_hook(get_hook(l, mode)) for l in features]
        
        count = 0
        with torch.no_grad():
            for batch in tqdm(loaders[mode]):
                img = batch[0] 
                model(img.to(device))
                count += 1
                if mode == 'bd' and count >= 16: break # BD 样本 1024 张足够
                
        # 移除钩子
        for h in handles: h.remove()

    # 保存结果
    for l in features:
        np.save(f'{save_dir}/{l}_clean.npy', np.concatenate(features[l]['clean'], axis=0))
        np.save(f'{save_dir}/{l}_bd.npy', np.concatenate(features[l]['bd'], axis=0))
    
    print(f"✅ {attack_name} 提取任务完美达成！")

# =========================================================
# 4. 主程序入口 (补全攻击路径)
# =========================================================
if __name__ == '__main__':
    # 请确保以下路径指向你 record 文件夹下的真实文件
    # 请确保以下路径指向你 record 文件夹下的真实文件
    attacks = {
        'badnets': './record/20260215_235930_badnet_attack_badnet_DvMv/attack_result.pt',
        'blended': './record/20260221_010831_blended_attack_blended_Aniv/attack_result.pt',
        'wanet': './record/20260221_013811_wanet_attack_wanet_CF3H/attack_result.pt',
        'sig': './record/sig_attack_1/attack_result.pt',
        'refool': './record/refool_attack_2/attack_result.pt',
        'inputaware': './record/inputaware_attack_1/attack_result.pt',
        'lira': './record/lira_attack_1/attack_result.pt',
        'bpp': './record/bpp_attack_1/attack_result.pt',
        'trojannn': './record/trojannn_attack_final/attack_result.pt',
        'badnet_all2all': './record/badnet_all2all_final/attack_result.pt' ,
        'blind': './record/blind_attack_final/attack_result.pt',  
        'ctrl': './record/ctrl_attack_final/attack_result.pt',
        'ftrojan': './record/ftrojan_attack_final/attack_result.pt' ,
 }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 UCAT 终端收割机启动 | 设备: {device}")

    for name, path in attacks.items():
        if os.path.exists(path):
            try:
                extract_final(name, path, device=device)
            except Exception as e:
                print(f"❌ 提取 {name} 失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"跳过 {name}，未找到模型文件。")

    print("\n🎉 全部 13 种攻击特征已存入 ./features_final/，准备运行 ucat_final_eval.py！")