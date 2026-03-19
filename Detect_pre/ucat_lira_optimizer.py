import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import os

def evaluate_l1(attack_name, layers=['layer2', 'layer3', 'layer4'], k=5):
    print(f"\n🎯 正在使用 L1-KNN (K={k}) 围剿 [ {attack_name} ]...")
    base_path = f'./features_final/{attack_name}'
    
    layer_scores_clean = []
    layer_scores_bd = []

    for layer in layers:
        clean = np.nan_to_num(np.load(f'{base_path}/{layer}_clean.npy'))
        bd = np.nan_to_num(np.load(f'{base_path}/{layer}_bd.npy'))
        
        # 1. 回归标准化 (不要 PCA)
        scaler = StandardScaler()
        c_scaled = scaler.fit_transform(clean)
        b_scaled = scaler.transform(bd)
        
        # 2. 切换到曼哈顿距离 (L1)
        # LIRA 的信号分散在多个维度，L1 距离比 L2 更能捕捉这种累积效应
        nn = NearestNeighbors(n_neighbors=k, n_jobs=-1, metric='manhattan')
        nn.fit(c_scaled)
        
        d_c, _ = nn.kneighbors(c_scaled)
        d_b, _ = nn.kneighbors(b_scaled)
        
        s_c = np.mean(d_c, axis=1)
        s_b = np.mean(d_b, axis=1)
        
        layer_scores_clean.append(s_c)
        layer_scores_bd.append(s_b)
        
        auc = roc_auc_score([0]*len(s_c) + [1]*len(s_b), np.concatenate([s_c, s_b]))
        print(f"  {layer} AUROC: {auc:.4f}")

    # 3. 简单的 Max-Pooling 融合
    all_c = np.max(layer_scores_clean, axis=0)
    all_b = np.max(layer_scores_bd, axis=0)
    
    final_auc = roc_auc_score([0]*len(all_c) + [1]*len(all_b), np.concatenate([all_c, all_b]))
    print(f"  >>> [L1 融合结果] AUROC: {final_auc:.4f}")
    return final_auc

if __name__ == '__main__':
    # 重点验证 LIRA 和 WaNet，这两类攻击最依赖度量方式
    for a in ['lira', 'wanet', 'bpp']:
        evaluate_l1(a, k=5)