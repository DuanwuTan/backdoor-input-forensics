import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# --- 配置区 ---
feature_dir = "./features_final"
output_csv = "ucat_grid_search_master_report.csv"

# 扫描所有可用的攻击（根据文件名自动提取）
all_files = [f for f in os.listdir(feature_dir) if f.endswith("_features.npy")]
attacks = sorted(list(set([f.split('_layer')[0] for f in all_files])))
layers = ["layer1", "layer2", "layer3", "layer4"]

# 超参数空间
ks = [1, 3, 5, 10, 20, 50]
metrics = ["manhattan", "euclidean"] # manhattan = L1, euclidean = L2

results = []

print(f"🔍 检测到 {len(attacks)} 种攻击任务，准备开始大规模网格搜索...")

for attack in attacks:
    print(f"\n🔥 正在分析攻击: {attack}")
    
    for layer in layers:
        # 1. 加载特征和标签
        feat_path = os.path.join(feature_dir, f"{attack}_{layer}_features.npy")
        lab_path = os.path.join(feature_dir, f"{attack}_{layer}_labels.npy")
        
        if not os.path.exists(feat_path) or not os.path.exists(lab_path):
            continue
            
        X = np.load(feat_path) # 形状通常是 (N, 5*Channels)
        y = np.load(lab_path) # 形状 (N,)
        
        # 2. 特征标准化 (KNN 必须步骤)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 3. 模拟无监督检测场景
        # 假设我们有少量干净样本作为“参考基准”（取前 1000 个干净样本）
        clean_indices = np.where(y == 0)[0]
        if len(clean_indices) < 1000:
            ref_size = len(clean_indices) // 2
        else:
            ref_size = 1000
            
        X_ref = X_scaled[clean_indices[:ref_size]]
        
        # 剩下的作为测试集（包含干净和后门）
        test_mask