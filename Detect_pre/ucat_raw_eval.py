import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
import os

def evaluate_layer(attack_name, layer, n_components=30):
    """对单个攻击的单个层进行评估，返回 AUROC 和 FPR@95TPR"""
    # 加载特征
    clean_path = f'./features_raw/{attack_name}/{layer}_clean.npy'
    bd_path = f'./features_raw/{attack_name}/{layer}_bd.npy'
    
    if not os.path.exists(clean_path) or not os.path.exists(bd_path):
        print(f"文件缺失: {attack_name} {layer}")
        return None, None
    
    clean = np.load(clean_path)
    bd = np.load(bd_path)
    
    # 处理可能的 NaN
    clean = np.nan_to_num(clean, nan=0.0)
    bd = np.nan_to_num(bd, nan=0.0)
    
    # 合并数据
    all_feats = np.vstack([clean, bd])
    
    # PCA 降维
    pca = PCA(n_components=min(n_components, all_feats.shape[0]))
    all_pca = pca.fit_transform(all_feats)
    
    clean_pca = all_pca[:len(clean)]
    bd_pca = all_pca[len(clean):]
    
    # IsolationForest 训练（在干净样本上）
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(clean_pca)
    
    # 获取异常分数（越负越异常）
    clean_scores = clf.decision_function(clean_pca)   # 正常样本得分高
    bd_scores = clf.decision_function(bd_pca)
    
    # 转换为“越正越异常”用于 AUROC
    scores = np.concatenate([-clean_scores, -bd_scores])
    labels = np.array([0]*len(clean_scores) + [1]*len(bd_scores))
    auroc = roc_auc_score(labels, scores)
    
    # 计算 FPR@95TPR
    # 阈值取后门样本得分的 5% 分位数（因为后门样本的 -score 应该更大，所以取高分阈值）
    # 更稳健：使用后门样本的异常分数（-bd_scores）的分位数
    bd_anomaly = -bd_scores
    thresh = np.percentile(bd_anomaly, 5)   # 5% 的后门样本低于此阈值，即 95% TPR 对应的阈值
    fpr = np.mean(-clean_scores < thresh)   # 干净样本中被判为异常的比例
    
    return auroc, fpr, -clean_scores, -bd_scores   # 返回分数以便融合

def fuse_scores(attack_name, layers):
    """对多个层进行最大值融合，返回融合后的 AUROC 和 FPR"""
    all_clean_scores = []
    all_bd_scores = []
    
    for layer in layers:
        res = evaluate_layer(attack_name, layer)
        if res[0] is None:
            continue
        _, _, clean_s, bd_s = res
        all_clean_scores.append(clean_s)
        all_bd_scores.append(bd_s)
    
    if not all_clean_scores:
        return None, None
    
    # 逐样本取最大值
    clean_max = np.max(all_clean_scores, axis=0)
    bd_max = np.max(all_bd_scores, axis=0)
    
    labels = np.array([0]*len(clean_max) + [1]*len(bd_max))
    scores = np.concatenate([clean_max, bd_max])
    auroc = roc_auc_score(labels, scores)
    
    thresh = np.percentile(bd_max, 5)
    fpr = np.mean(clean_max < thresh)
    
    return auroc, fpr

def main():
    attacks = ['badnets', 'blended', 'wanet', 'sig', 'refool', 'inputaware']
    layers = ['layer3', 'layer4']
    
    print("=== 各层单独结果 ===")
    for attack in attacks:
        print(f"\n攻击: {attack}")
        for layer in layers:
            auroc, fpr, _, _ = evaluate_layer(attack, layer)
            if auroc is not None:
                print(f"  {layer}: AUROC={auroc:.4f}, FPR@95TPR={fpr:.4f}")
    
    print("\n=== 融合结果 (layer3+layer4 最大值) ===")
    for attack in attacks:
        auroc, fpr = fuse_scores(attack, layers)
        if auroc is not None:
            print(f"{attack}: AUROC={auroc:.4f}, FPR@95TPR={fpr:.4f}")

if __name__ == '__main__':
    main()