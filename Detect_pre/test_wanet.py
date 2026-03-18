import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import os

def test_wanet_special(layer='layer2'): # WaNet 在 layer2 表现稍好
    path = './features_v3/wanet'
    clean = np.load(f'{path}/{layer}_clean.npy')
    bd = np.load(f'{path}/{layer}_bd.npy')
    
    C = clean.shape[1] // 4 # 得到通道数
    
    # 实验 1：全特征 (V3)
    # 实验 2：纯梯度特征 (后一半维度)
    # 实验 3：不带 PCA
    
    # 我们直接测试“纯梯度 + OCSVM”
    grad_only_clean = clean[:, 2*C:] 
    grad_only_bd = bd[:, 2*C:]
    
    scaler = StandardScaler()
    c_scaled = scaler.fit_transform(grad_only_clean)
    b_scaled = scaler.transform(grad_only_bd)
    
    # 测试不同检测器
    detectors = {
        "IF_30": IsolationForest(contamination=0.05, random_state=0),
        "OCSVM_RBF": OneClassSVM(kernel='rbf', nu=0.05),
    }
    
    print(f"--- WaNet 专项测试 ({layer}) ---")
    for name, clf in detectors.items():
        clf.fit(c_scaled)
        if hasattr(clf, "decision_function"):
            c_score = -clf.decision_function(c_scaled)
            b_score = -clf.decision_function(b_scaled)
        else:
            c_score = -clf.score_samples(c_scaled)
            b_score = -clf.score_samples(b_scaled)
            
        auc = roc_auc_score([0]*len(c_score) + [1]*len(b_score), np.concatenate([c_score, b_score]))
        print(f"检测器: {name}, AUROC: {auc:.4f}")

if __name__ == '__main__':
    for l in ['layer2', 'layer3', 'layer4']:
        test_wanet_special(l)