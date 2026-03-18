import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import os
from sklearn.preprocessing import StandardScaler

def evaluate_layer(attack_name, layer, base_path='./features_v2', n_components=30):
    """对单个攻击的单个层进行评估，返回 AUROC 和 FPR@95TPR"""
    clean_path = os.path.join(base_path, attack_name, f'{layer}_clean.npy')
    bd_path = os.path.join(base_path, attack_name, f'{layer}_bd.npy')
    
    if not os.path.exists(clean_path) or not os.path.exists(bd_path):
        print(f"文件缺失: {attack_name} {layer}")
        return None, None
    
    clean = np.load(clean_path)
    bd = np.load(bd_path)
    
    # 处理可能的 NaN
    clean = np.nan_to_num(clean, nan=0.0)
    bd = np.nan_to_num(bd, nan=0.0)
    
    # 只取通道均值部分（假设特征顺序是 [channel_mean, spatial_std, gini]）
     
    # # 1. 提取通道均值
    # C = clean.shape[1] - 2
    # clean_m = clean[:, :C]
    # bd_m = bd[:, :C]
    # 使用完整 v2 特征（不截取）
    clean_m = clean   # 完整特征
    bd_m = bd
    # 2. 【修复关键】PCA 必须只在干净样本上 fit
    scaler = StandardScaler()
    clean_scaled = scaler.fit_transform(clean_m)
    bd_scaled = scaler.transform(bd_m)
    pca = PCA(n_components=min(n_components, clean_m.shape[1], clean_m.shape[0]-1))
    clean_pca = pca.fit_transform(clean_scaled)  # 使用标准化后的数据
    bd_pca = pca.transform(bd_scaled)            # 使用标准化后的数据
    # 3. IsolationForest 保持在干净样本上训练
    clf = IsolationForest(contamination=0.05, random_state=0) # 建议对齐 v1 参数
    clf.fit(clean_pca)
    
    # 4. 分数计算（确保负号正确）
    # decision_function 越负越异常
    clean_scores = clf.decision_function(clean_pca) 
    bd_scores = clf.decision_function(bd_pca)
    
    # 为了计算 AUC，我们需要“越高越异常”的分数
    clean_anomaly = -clean_scores
    bd_anomaly = -bd_scores
    
    # ... 后续计算 AUC 和 FPR 的逻辑 ...
    
    scores = np.concatenate([clean_anomaly, bd_anomaly])
    labels = np.array([0]*len(clean_anomaly) + [1]*len(bd_anomaly))
    auroc = roc_auc_score(labels, scores)
  # 计算 FPR@95TPR
    # 逻辑：找到后门样本的第 5 百分位数，确保 95% 的后门样本得分高于此阈值
    thresh = np.percentile(bd_anomaly, 5)   
    # FPR 是干净样本中得分超过该阈值的比例
    fpr = np.mean(clean_anomaly >= thresh)
    return auroc, fpr, clean_anomaly, bd_anomaly

def fuse_scores(attack_name, layers, base_path='./features_v2'):
    """对多个层进行最大值融合，返回融合后的 AUROC 和 FPR"""
    all_clean = []
    all_bd = []
    
    for layer in layers:
        res = evaluate_layer(attack_name, layer, base_path)
        if res[0] is None:
            continue
        _, _, clean_s, bd_s = res
        all_clean.append(clean_s)
        all_bd.append(bd_s)
    
    if not all_clean:
        return None, None
    
    # 逐样本取最大值
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
    
    print("=== 各层单独结果（新特征 v2）===")
    for attack in attacks:
        print(f"\n攻击: {attack}")
        for layer in layers:
            try:
                auroc, fpr, _, _ = evaluate_layer(attack, layer)
                if auroc is not None:
                    print(f"  {layer}: AUROC={auroc:.4f}, FPR@95TPR={fpr:.4f}")
            except Exception as e:
                print(f"  {layer}: 错误 - {e}")
    
    print("\n=== 融合结果 (layer1~layer4 最大值) ===")
    for attack in attacks:
        auroc, fpr = fuse_scores(attack, layers)
        if auroc is not None:
            print(f"{attack}: AUROC={auroc:.4f}, FPR@95TPR={fpr:.4f}")

if __name__ == '__main__':
    main()