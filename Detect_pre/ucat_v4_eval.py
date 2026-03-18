import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import os

def evaluate_v4_knn(attack_name, layer, k=10):
    """使用 KNN 评估 V4 质心特征"""
    path = f'./features_v4/{attack_name}'
    clean_path = f'{path}/{layer}_clean.npy'
    bd_path = f'{path}/{layer}_bd.npy'
    
    if not os.path.exists(clean_path):
        return None
    
    clean = np.load(clean_path)
    bd = np.load(bd_path)
    
    # 1. 预处理：修复 NaN 并标准化
    clean = np.nan_to_num(clean)
    bd = np.nan_to_num(bd)
    
    scaler = StandardScaler()
    # 只根据干净样本拟合标准化参数
    c_scaled = scaler.fit_transform(clean)
    b_scaled = scaler.transform(bd)
    
    # 2. KNN 检测器
    # 我们计算每个点到干净样本集合中最近 k 个点的平均距离
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean', n_jobs=-1)
    nn.fit(c_scaled)
    
    # 计算干净样本自身的异常分数（留一法距离）
    dist_c, _ = nn.kneighbors(c_scaled)
    score_c = np.mean(dist_c, axis=1)
    
    # 计算后门样本的异常分数
    dist_b, _ = nn.kneighbors(b_scaled)
    score_b = np.mean(dist_b, axis=1)
    
    # 3. 计算指标
    y_true = np.concatenate([np.zeros(len(score_c)), np.ones(len(score_b))])
    y_scores = np.concatenate([score_c, score_b])
    
    auroc = roc_auc_score(y_true, y_scores)
    
    # 计算 FPR@95TPR
    thresh = np.percentile(score_b, 5) # 95% 的后门样本高于此阈值
    fpr = np.mean(score_c >= thresh)
    
    return auroc, fpr

if __name__ == '__main__':
    print("=== UCAT V4 (空间质心) 评估结果 ===")
    attack = 'wanet'
    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    
    results = []
    for layer in layers:
        res = evaluate_v4_knn(attack, layer)
        if res:
            auc, fpr = res
            print(f"层 {layer}: AUROC = {auc:.4f}, FPR@95TPR = {fpr:.4f}")
            results.append((auc, fpr))
    
    if results:
        max_auc = max([r[0] for r in results])
        print(f"\n>>> WaNet 最佳检测 AUROC: {max_auc:.4f}")