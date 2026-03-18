import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# 只看 WaNet 最强的 Layer 2，且只用质心特征 (维度 C*2)
def check_wanet_pure():
    path = './features_final/wanet'
    # 提取第 3、4 维 (对应 centroid_y, centroid_x)
    clean = np.load(f'{path}/layer2_clean.npy')
    bd = np.load(f'{path}/layer2_bd.npy')
    
    C = clean.shape[1] // 5 # 我们拼了 5 组特征
    # 只取质心特征部分 (2*C 到 4*C)
    clean_pure = clean[:, 2*C:4*C]
    bd_pure = bd[:, 2*C:4*C]
    
    scaler = StandardScaler()
    c_s = scaler.fit_transform(clean_pure)
    b_s = scaler.transform(bd_pure)
    
    nn = NearestNeighbors(n_neighbors=10).fit(c_s)
    d_c = np.mean(nn.kneighbors(c_s)[0], axis=1)
    d_b = np.mean(nn.kneighbors(b_s)[0], axis=1)
    
    auc = roc_auc_score([0]*len(d_c) + [1]*len(d_b), np.concatenate([d_c, d_b]))
    thresh = np.percentile(d_b, 5)
    fpr = np.mean(d_c >= thresh)
    
    print(f"WaNet Layer 2 (纯质心特征) -> AUROC: {auc:.4f}, FPR@95TPR: {fpr:.4f}")

check_wanet_pure()