import sys
sys.path.append("D:\\project_backdoor\\BackdoorBench")

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
from utils.save_load_attack import load_attack_result

def extract_features_from_layer(dataset, model, layer_name, device, max_samples=500):
    """
    从指定层提取特征图，展平后作为特征向量
    """
    target_layer = getattr(model, layer_name)
    activations = []
    
    def hook_fn(module, input, output):
        # output shape: (batch, C, H, W)
        activations.append(output.detach().cpu().numpy())
    
    handle = target_layer.register_forward_hook(hook_fn)
    
    features = []
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        img = sample[0].unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(img)
        # 取出当前样本的激活
        feat_map = activations.pop()  # 确保每次只取一个
        # 展平
        feat_vec = feat_map.flatten()
        features.append(feat_vec)
        if (i+1) % 100 == 0:
            print(f"    {layer_name}: 已处理 {i+1}/{max_samples}")
    
    handle.remove()
    return np.array(features)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 六种攻击的路径和名称
    attacks = {
        'badnets': './record/20260215_235930_badnet_attack_badnet_DvMv/attack_result.pt',
        'blended': './record/20260221_010831_blended_attack_blended_Aniv/attack_result.pt',
        'wanet': './record/20260221_013811_wanet_attack_wanet_CF3H/attack_result.pt',
        'sig': './record/sig_attack_1/attack_result.pt',
        'refool': './record/refool_attack_2/attack_result.pt',
        'inputaware': './record/inputaware_attack_1/attack_result.pt',
    }

    layers = ['layer3', 'layer4']
    max_samples = 500

    for attack_name, model_path in attacks.items():
        print(f"\n========== 处理攻击: {attack_name} ==========")
        attack_result = load_attack_result(model_path)
        model_state = attack_result['model']

        # 重建模型
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(512, 10)  # CIFAR-10
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()

        clean_dataset = attack_result['clean_test']
        poison_dataset = attack_result['bd_test']

        # 创建保存目录
        save_dir = f'./features_raw/{attack_name}'
        os.makedirs(save_dir, exist_ok=True)

        for layer in layers:
            print(f"\n--- 提取 {layer} 特征 ---")
            
            # 提取干净样本
            clean_feats = extract_features_from_layer(clean_dataset, model, layer, device, max_samples)
            # 提取后门样本
            poison_feats = extract_features_from_layer(poison_dataset, model, layer, device, max_samples)

            # 打印形状以便确认维度
            print(f"  干净特征形状: {clean_feats.shape}, 后门特征形状: {poison_feats.shape}")
            
            # 保存
            np.save(os.path.join(save_dir, f'{layer}_clean.npy'), clean_feats)
            np.save(os.path.join(save_dir, f'{layer}_bd.npy'), poison_feats)
            print(f"  {layer} 特征保存完成")

if __name__ == '__main__':
    main()