import sys
sys.path.append("D:\\project_backdoor\\BackdoorBench")

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
from utils.save_load_attack import load_attack_result

def gini_coefficient(x):
    x = np.sort(np.abs(x))
    n = len(x)
    index = np.arange(1, n+1)
    if np.sum(x) == 0:
        return 0.0
    return (2 * np.sum(index * x) / (n * np.sum(x))) - (n+1)/n

def extract_features_v2(dataset, model, layer_name, device, max_samples=500):
    target_layer = getattr(model, layer_name)
    activations = []
    
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu().numpy())
    
    handle = target_layer.register_forward_hook(hook_fn)
    
    features = []
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        img = sample[0].unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(img)
        feat_map = activations.pop()  # (1, C, H, W)
        # 1. 通道均值
        channel_mean = feat_map.mean(axis=(2,3)).flatten()
        # 2. 空间标准差（跨通道标准差图的空间均值）
        spatial_std_map = feat_map.std(axis=1)  # (1, H, W)
        spatial_std = spatial_std_map.mean()
        # 3. 基尼系数
        gini = gini_coefficient(channel_mean)
        # 拼接
        feat_vec = np.concatenate([channel_mean, [spatial_std, gini]])
        features.append(feat_vec)
        
        if (i+1) % 100 == 0:
            print(f"    {layer_name}: 已处理 {i+1}/{max_samples}")
    
    handle.remove()
    return np.array(features)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    attacks = {
        'badnets': './record/20260215_235930_badnet_attack_badnet_DvMv/attack_result.pt',
        'blended': './record/20260221_010831_blended_attack_blended_Aniv/attack_result.pt',
        'wanet': './record/20260221_013811_wanet_attack_wanet_CF3H/attack_result.pt',
        'sig': './record/sig_attack_1/attack_result.pt',
        'refool': './record/refool_attack_2/attack_result.pt',
        'inputaware': './record/inputaware_attack_1/attack_result.pt',
    }
    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    max_samples = 500

    for attack_name, model_path in attacks.items():
        print(f"\n========== 处理攻击: {attack_name} ==========")
        attack_result = load_attack_result(model_path)
        model_state = attack_result['model']
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, 10)
        model.load_state_dict(model_state)
        model = model.to(device).eval()

        clean_dataset = attack_result['clean_test']
        bd_dataset = attack_result['bd_test']

        for layer in layers:
            print(f"\n--- 提取 {layer} 特征 ---")
            clean_feats = extract_features_v2(clean_dataset, model, layer, device, max_samples)
            bd_feats = extract_features_v2(bd_dataset, model, layer, device, max_samples)

            save_dir = f'./features_v2/{attack_name}'
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f'{layer}_clean.npy'), clean_feats)
            np.save(os.path.join(save_dir, f'{layer}_bd.npy'), bd_feats)
            print(f"    {layer} 保存完成: 干净 {clean_feats.shape}, 后门 {bd_feats.shape}")

if __name__ == '__main__':
    main()