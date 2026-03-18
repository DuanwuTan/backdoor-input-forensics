import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import os
from sklearn.preprocessing import StandardScaler

def evaluate_layer(attack_name, layer, base_path='./features_v3', n_components=30):
    """对单个攻击的单个层进行评估"""
    clean_path = os.path.join(base_path, attack_name, f'{layer}_clean.npy')
    bd_path = os.path.join(base_path, attack_name, f'{layer}_bd.npy')
    
    if not os.path.exists(clean_path) or not os.path.exists(bd_path):
        return None, None, None, None
    
    clean = np.load(clean_path)
    bd = np.load(bd_path)
    
    # 处理 NaN (std 在 1x1 卷积核下可能产生 NaN)
    clean = np.nan_to_num(clean, nan=0.0)
    bd = np.nan_to_num(bd, nan=0.0)
    
    # 使用完整 v3 特征 (Mean, Gini, Grad_H, Grad_V)
    # 维度应该是 C*4
    scaler = StandardScaler()
    clean_scaled = scaler.fit_transform(clean)
    bd_scaled = scaler.transform(bd)
    
    # PCA 降维，尝试保留核心异常信号
    n_feat = min(n_components, clean.shape[1], clean.shape[0]-1)
    pca = PCA(n_components=n_feat)
    clean_pca = pca.fit_transform(clean_scaled)
    bd_pca = pca.transform(bd_scaled)
    
    # 无监督检测器
    clf = IsolationForest(contamination=0.05, random_state=0)
    clf.fit(clean_pca)
    
    # 计算异常得分 (越高越异常)
    clean_anomaly = -clf.decision_function(clean_pca)
    bd_anomaly = -clf.decision_function(bd_pca)
    
    # 计算 AUROC
    scores = np.concatenate([clean_anomaly, bd_anomaly])
    labels = np.array([0]*len(clean_anomaly) + [1]*len(bd_anomaly))
    auroc = roc_auc_score(labels, scores)
    
    # 计算 FPR@95TPR (修正后的逻辑)
    thresh = np.percentile(bd_anomaly, 5)
    fpr = np.mean(clean_anomaly >= thresh)
    
    return auroc, fpr, clean_anomaly, bd_anomaly

def fuse_scores(attack_name, layers, base_path='./features_v3'):
    all_clean = []
    all_bd = []
    for layer in layers:
        res = evaluate_layer(attack_name, layer, base_path)
        if res[0] is not None:
            _, _, clean_s, bd_s = res
            all_clean.append(clean_s)
            all_bd.append(bd_s)
    
    if not all_clean: return None, None
    clean_max = np.max(all_clean, axis=0)
    bd_max = np.max(all_bd, axis=0)
    labels = np.array([0]*len(clean_max) + [1]*len(bd_max))
    scores = np.concatenate([clean_max, bd_max])
    auroc = roc_auc_score(labels, scores)
    thresh = np.percentile(bd_max, 5)
    fpr = np.mean(clean_max >= thresh)
    return auroc, fpr

def main():
    attacks = ['badnets', 'blended', 'wanet', 'sig', 'refool', 'inputaware']
    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    
    print("=== V3 增强特征评估结果 (均值+基尼+空间梯度) ===")
    for attack in attacks:
        print(f"\n攻击: {attack}")
        for layer in layers:
            auroc, fpr, _, _ = evaluate_layer(attack, layer)
            if auroc is not None:
                print(f"  {layer}: AUROC={auroc:.4f}, FPR@95TPR={fpr:.4f}")
        
        # 融合结果
        f_auroc, f_fpr = fuse_scores(attack, layers)
        if f_auroc is not None:
            print(f"  [FUSED]: AUROC={f_auroc:.4f}, FPR@95TPR={f_fpr:.4f}")

if __name__ == '__main__':
    main()