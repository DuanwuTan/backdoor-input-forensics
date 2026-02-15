import sys
sys.path.append("D:\\project_backdoor\\BackdoorBench")
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from utils.save_load_attack import load_attack_result

def extract_features(dataset, model, device, max_samples=500):
    """
    提取 model.layer4 的输出特征图，展平后作为特征向量
    """
    features = []
    target_layer = model.layer4   # 整个 layer4 模块，输出形状 [B, 512, 4, 4]

    activations = []
    def forward_hook(module, input, output):
        activations.append(output.detach().cpu().numpy())

    handle = target_layer.register_forward_hook(forward_hook)

    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        img = sample[0].unsqueeze(0).to(device)

        # 前向传播（触发 hook）
        with torch.no_grad():
            _ = model(img)

        # 获取特征图
        feat_map = activations[0]   # (1, 512, 4, 4)
        activations.clear()

        # 展平为特征向量
        feat_vec = feat_map.flatten()
        features.append(feat_vec)

        if (i+1) % 100 == 0:
            print(f"已处理 {i+1}/{max_samples}")

    handle.remove()
    return np.array(features)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 新模型路径
    model_path = "./record/20260215_235930_badnet_attack_badnet_DvMv/attack_result.pt"
    attack_result = load_attack_result(model_path)

    # 重建模型（与训练结构一致）
    model_state = attack_result['model']
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 10)   # CIFAR-10
    model.load_state_dict(model_state)
    model = model.to(device)
    model.eval()

    # 获取数据集
    clean_dataset = attack_result['clean_test']
    poison_dataset = attack_result['bd_test']

    print("提取干净样本特征...")
    clean_feats = extract_features(clean_dataset, model, device, max_samples=500)

    print("提取后门样本特征...")
    poison_feats = extract_features(poison_dataset, model, device, max_samples=500)

    # 保存特征
    np.save('features_badnet_clean.npy', clean_feats)
    np.save('features_badnet_poison.npy', poison_feats)
    print(f"特征保存完成！")
    print(f"干净特征形状: {clean_feats.shape}, 后门特征形状: {poison_feats.shape}")

if __name__ == '__main__':
    main()