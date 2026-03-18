import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import os

def evaluate_final(attack_name, layers=['layer2', 'layer3', 'layer4']):
    print(f"\n评估攻击: {attack_name}")
    base_path = f'./features_final/{attack_name}'
    
    layer_scores_clean = []
    layer_scores_bd = []

    for layer in layers:
        clean = np.load(f'{base_path}/{layer}_clean.npy')
        bd = np.load(f'{base_path}/{layer}_bd.npy')
        
        # 标准化
        scaler = StandardScaler()
        c_scaled = scaler.fit_transform(np.nan_to_num(clean))
        b_scaled = scaler.transform(np.nan_to_num(bd))
        
        # KNN 检测
        nn = NearestNeighbors(n_neighbors=10, n_jobs=-1)
        nn.fit(c_scaled)
        
        d_c, _ = nn.kneighbors(c_scaled)
        d_b, _ = nn.kneighbors(b_scaled)
        
        layer_scores_clean.append(np.mean(d_c, axis=1))
        layer_scores_bd.append(np.mean(d_b, axis=1))
        
        auc = roc_auc_score([0]*len(d_c) + [1]*len(d_b), np.concatenate([layer_scores_clean[-1], layer_scores_bd[-1]]))
        print(f"  {layer} AUROC: {auc:.4f}")

    # 多层融合 (Max-Pooling 策略)
    all_c = np.max(layer_scores_clean, axis=0)
    all_b = np.max(layer_scores_bd, axis=0)
    
    final_auc = roc_auc_score([0]*len(all_c) + [1]*len(all_b), np.concatenate([all_c, all_b]))
    thresh = np.percentile(all_b, 5)
    final_fpr = np.mean(all_c >= thresh)
    
    print(f"  >>> [融合结果] AUROC: {final_auc:.4f}, FPR@95TPR: {final_fpr:.4f}")
    return final_auc, final_fpr

if __name__ == '__main__':
    attacks = ['badnets', 'blended', 'wanet', 'sig', 'refool', 'inputaware']
    for a in attacks:
        try:
            evaluate_final(a)
        except Exception as e:
            print(f"评估 {a} 失败: {e}")