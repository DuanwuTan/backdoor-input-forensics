import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def rescue_wanet():
    path = './features_final/wanet'
    # 加载 Layer 2 质心
    clean = np.load(f'{path}/layer2_clean.npy')
    bd = np.load(f'{path}/layer2_bd.npy')
    C = clean.shape[1] // 5
    clean_pure = clean[:, 2*C:4*C]
    bd_pure = bd[:, 2*C:4*C]
    
    # 1. 标准化
    scaler = StandardScaler()
    c_s = scaler.fit_transform(clean_pure)
    b_s = scaler.transform(bd_pure)
    
    # 2. 强力降维 (只保留最重要的 10 个方向，过滤噪声)
    pca = PCA(n_components=10)
    c_pca = pca.fit_transform(c_s)
    b_pca = pca.transform(b_s)
    
    # 3. 使用隔离林 (它比 KNN 对“分布偏移”更敏感)
    clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    clf.fit(c_pca)
    
    score_c = -clf.decision_function(c_pca)
    score_b = -clf.decision_function(b_pca)
    
    auc = roc_auc_score([0]*len(score_c) + [1]*len(score_b), np.concatenate([score_c, score_b]))
    thresh = np.percentile(score_b, 5)
    fpr = np.mean(score_c >= thresh)
    
    print(f"WaNet 拯救行动 -> AUROC: {auc:.4f}, FPR@95TPR: {fpr:.4f}")

rescue_wanet()