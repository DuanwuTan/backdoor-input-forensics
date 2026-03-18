import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

def load_and_clean(path, layer):
    clean = np.load(f'{path}/{layer}_clean.npy')
    bd = np.load(f'{path}/{layer}_bd.npy')
    
    # 1. 处理 NaN 和无穷大
    clean = np.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0)
    bd = np.nan_to_num(bd, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. 剔除全为常数的列 (防止 StandardScaler 报错)
    # 只根据干净样本判断哪些列是有意义的
    variances = np.var(clean, axis=0)
    useful_cols = variances > 1e-9
    
    clean = clean[:, useful_cols]
    bd = bd[:, useful_cols]
    
    return clean, bd

def test_wanet_optimized(layer='layer2'):
    path = './features_v3/wanet'
    if not os.path.exists(f'{path}/{layer}_clean.npy'): return
    
    clean, bd = load_and_clean(path, layer)
    
    # 标准化
    scaler = StandardScaler()
    c_scaled = scaler.fit_transform(clean)
    b_scaled = scaler.transform(bd)
    
    # PCA 降维 (保留 95% 的方差，这对 WaNet 很重要，不能降得太狠)
    pca = PCA(n_components=0.95)
    c_pca = pca.fit_transform(c_scaled)
    b_pca = pca.transform(b_scaled)
    
    print(f"\n--- WaNet 深度测试 ({layer}) ---")
    print(f"原始特征维度: {clean.shape[1]}, PCA后维度: {c_pca.shape[1]}")
    
    detectors = {
        "IF (100 trees)": IsolationForest(contamination=0.05, random_state=42),
        "OCSVM (RBF)": OneClassSVM(kernel='rbf', nu=0.05, gamma='scale'),
        "OCSVM (Sigmoid)": OneClassSVM(kernel='sigmoid', nu=0.05)
    }
    
    for name, clf in detectors.items():
        clf.fit(c_pca)
        # 获取异常得分
        c_score = -clf.decision_function(c_pca)
        b_score = -clf.decision_function(b_pca)
            
        auc = roc_auc_score([0]*len(c_score) + [1]*len(b_score), np.concatenate([c_score, b_score]))
        print(f"[{name}] AUROC: {auc:.4f}")

if __name__ == '__main__':
    for l in ['layer2', 'layer3', 'layer4']:
        try:
            test_wanet_optimized(l)
        except Exception as e:
            print(f"层 {l} 评估失败: {e}")