import sys
sys.path.append("D:\\project_backdoor\\BackdoorBench")

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from utils.save_load_attack import load_attack_result

def extract_flat_features(dataset, model, layer_name, device, max_samples=500):
    """提取指定层的展平特征（原始空间激活）"""
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
        features.append(feat_map.flatten())
        
        if (i+1) % 100 == 0:
            print(f"    {layer_name}: 已处理 {i+1}/{max_samples}")
    
    handle.remove()
    return np.array(features)

def evaluate_layer_lr(clean_feats, bd_feats, n_splits=5):
    """使用逻辑回归5折交叉验证计算AUC"""
    X = np.vstack([clean_feats, bd_feats])
    y = np.array([0]*len(clean_feats) + [1]*len(bd_feats))
    
    # 可选：PCA降维（如果维度太高导致计算慢，可以启用）
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=min(50, X.shape[0]))
    # X = pca.fit_transform(X)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, solver='lbfgs')
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], y_pred))
    return np.mean(aucs), np.std(aucs)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 六种攻击的路径（请根据实际路径确认）
    attacks = {
        'badnets': './record/20260215_235930_badnet_attack_badnet_DvMv/attack_result.pt',
        'blended': './record/20260221_010831_blended_attack_blended_Aniv/attack_result.pt',
        'wanet': './record/20260221_013811_wanet_attack_wanet_CF3H/attack_result.pt',
        'sig': './record/sig_attack_1/attack_result.pt',
        'refool': './record/refool_attack_2/attack_result.pt',
        'inputaware': './record/inputaware_attack_1/attack_result.pt',
    }

    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    max_samples = 500  # 可调整为1000以更稳定，但时间会长一些

    results = {}

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
        bd_dataset = attack_result['bd_test']

        best_auc = 0.0
        best_layer = None
        for layer in layers:
            print(f"\n--- 提取 {layer} 特征 ---")
            clean_feats = extract_flat_features(clean_dataset, model, layer, device, max_samples)
            bd_feats = extract_flat_features(bd_dataset, model, layer, device, max_samples)

            auc_mean, auc_std = evaluate_layer_lr(clean_feats, bd_feats)
            print(f"    {layer}: AUC = {auc_mean:.4f} ± {auc_std:.4f}")

            if auc_mean > best_auc:
                best_auc = auc_mean
                best_layer = layer

        results[attack_name] = {'best_auc': best_auc, 'best_layer': best_layer}
        print(f"\n{attack_name} 最佳层: {best_layer}, AUC = {best_auc:.4f}")

    print("\n========== 最终结果 ==========")
    for attack, res in results.items():
        print(f"{attack}: 最佳AUC = {res['best_auc']:.4f} (层 {res['best_layer']})")

if __name__ == '__main__':
    main()