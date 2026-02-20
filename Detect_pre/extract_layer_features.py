import sys
sys.path.append("D:\\project_backdoor\\BackdoorBench")  # 确保能找到 utils

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from utils.save_load_attack import load_attack_result

def extract_features_from_layer(dataset, model, layer_name, device, max_samples=500):
    """
    从指定层提取特征图（展平后）
    Args:
        dataset: 数据集，每个元素为 (img, label)
        model: 模型
        layer_name: 字符串，如 'layer1', 'layer2', 'layer3', 'layer4'
        device: 设备
        max_samples: 最大样本数
    Returns:
        numpy array of shape (max_samples, feature_dim)
    """
    target_layer = getattr(model, layer_name)   # 获取模型中的该层
    activations = []
    
    def forward_hook(module, input, output):
        # output 是 (batch, C, H, W)，我们保存它
        activations.append(output.detach().cpu().numpy())
    
    handle = target_layer.register_forward_hook(forward_hook)
    
    features = []
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        # 从 sample 中取出图像（注意：dataset 返回的可能是 (img, label) 或 (img, label, index) 等）
        # 我们只取第一个元素作为图像
        img = sample[0].unsqueeze(0).to(device)   # (1,3,H,W)
        
        with torch.no_grad():
            _ = model(img)   # 前向传播，hook 会被触发
        
        feat_map = activations[0]
        if i == 0:  # 只在第一个样本时打印
            print(f"feat_map shape: {feat_map.shape}")
        feat_vec = feat_map.flatten()   # 展平为一维
        features.append(feat_vec)
        
        if (i+1) % 100 == 0:
            print(f"已处理 {i+1}/{max_samples} 样本，来自层 {layer_name}")
    
    handle.remove()
    return np.array(features)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 模型路径（请确认这是你训练好的后门模型）
    model_path = "./record/20260215_235930_badnet_attack_badnet_DvMv/attack_result.pt"
    attack_result = load_attack_result(model_path)

    # 重建模型（与训练结构一致）
    model_state = attack_result['model']
    model = models.resnet18(weights=None)
    # 适配 CIFAR-10 的修改（）
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    # 获取数据集
    clean_dataset = attack_result['clean_test']
    poison_dataset = attack_result['bd_test']

    # 定义要提取的层列表
    layers = ['layer1', 'layer2', 'layer3', 'layer4']

    for layer in layers:
        print(f"\n===== 提取 {layer} 特征 =====")
        clean_feats = extract_features_from_layer(clean_dataset, model, layer, device, max_samples=500)
        poison_feats = extract_features_from_layer(poison_dataset, model, layer, device, max_samples=500)

        # 保存特征文件
        np.save(f'features_badnet_clean_{layer}.npy', clean_feats)
        np.save(f'features_badnet_poison_{layer}.npy', poison_feats)

        print(f"{layer} 特征保存完成！")
        print(f"干净特征形状: {clean_feats.shape}, 后门特征形状: {poison_feats.shape}")

if __name__ == '__main__':
    main()