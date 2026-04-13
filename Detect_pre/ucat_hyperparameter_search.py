import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import time

base_dir = r"D:\project_backdoor\BackdoorBench"
feature_root = os.path.join(base_dir, "features_final")
output_file = os.path.join(base_dir, "UCAT_GridSearch_Final_Corrected.csv")

attack_dirs = [d for d in os.listdir(feature_root) if os.path.isdir(os.path.join(feature_root, d))]
ks = [1, 3, 5, 10, 20, 50]
metrics = ["manhattan", "euclidean"]

with open(output_file, "w") as f:
    f.write("Attack,Layer,K,Metric,Raw_AUC,Corrected_AUC,Direction\n")

for attack in attack_dirs:
    attack_path = os.path.join(feature_root, attack)
    print(f"\n🚀 正在分析攻击家族: {attack}")
    
    for layer_idx in range(1, 5):
        layer_name = f"layer{layer_idx}"
        clean_file = os.path.join(attack_path, f"{layer_name}_clean.npy")
        bd_file = os.path.join(attack_path, f"{layer_name}_bd.npy")
        
        if os.path.exists(clean_file) and os.path.exists(bd_file):
            try:
                X_clean = np.load(clean_file)
                X_bd = np.load(bd_file)
                
                # 标准化
                scaler = StandardScaler()
                split_idx = len(X_clean) // 2
                ref_X = scaler.fit_transform(X_clean[:split_idx])
                test_X = scaler.transform(np.concatenate([X_clean[split_idx:], X_bd]))
                test_y = np.concatenate([np.zeros(len(X_clean)-split_idx), np.ones(len(X_bd))])

                for m in metrics:
                    for k in ks:
                        if k > len(ref_X): continue
                        
                        nn = NearestNeighbors(n_neighbors=k, metric=m, n_jobs=-1)
                        nn.fit(ref_X)
                        distances, _ = nn.kneighbors(test_X)
                        scores = distances.mean(axis=1)
                        
                        raw_auc = roc_auc_score(test_y, scores)
                        
                        # 【核心逻辑】：识别分离能力
                        # 如果 raw_auc < 0.5，说明后门样本是“向内塌缩”的异常
                        # 如果 raw_auc > 0.5，说明后门样本是“向外离群”的异常
                        corrected_auc = max(raw_auc, 1 - raw_auc)
                        direction = "Inward" if raw_auc < 0.5 else "Outward"
                        
                        print(f"  ✅ {layer_name} | K={k:2d} | AUC={corrected_auc:.4f} ({direction})")
                        
                        with open(output_file, "a") as f_out:
                            f_out.write(f"{attack},{layer_name},{k},{m},{raw_auc:.4f},{corrected_auc:.4f},{direction}\n")
                            
            except Exception as e:
                print(f"  ❌ 出错: {e}")

print(f"\n🏆 全量搜索已启动，数据将保存在: {output_file}")