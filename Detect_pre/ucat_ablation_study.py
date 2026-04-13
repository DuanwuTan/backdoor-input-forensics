import os
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# --- 配置 ---
base_dir = r"D:\project_backdoor\BackdoorBench"
feature_root = os.path.join(base_dir, "features_final")
output_file = os.path.join(base_dir, "UCAT_Full_Group_Ablation.csv")

# 自动扫描所有攻击目录
attack_dirs = [d for d in os.listdir(feature_root) if os.path.isdir(os.path.join(feature_root, d))]

# 定义 5 类特征组
feature_groups = {
    "Mean": 0,
    "Gini": 1,
    "Std": 2,
    "Centroid": 3,
    "PAPR": 4
}

with open(output_file, "w") as f:
    f.write("Attack,Layer,Dropped_Group,AUC\n")

print(f"🔍 监测到 {len(attack_dirs)} 种攻击，准备开始全量组消融实验...")

for i, attack in enumerate(attack_dirs):
    attack_path = os.path.join(feature_root, attack)
    print(f"\n🚀 [{i+1}/{len(attack_dirs)}] 处理攻击: {attack}")
    
    for layer_idx in range(1, 5):
        layer_name = f"layer{layer_idx}"
        cf = os.path.join(attack_path, f"{layer_name}_clean.npy")
        bf = os.path.join(attack_path, f"{layer_name}_bd.npy")
        
        if os.path.exists(cf) and os.path.exists(bf):
            try:
                X_c, X_b = np.load(cf), np.load(bf)
                num_dim = X_c.shape[1]
                
                # 特征是按类别拼接的，每类占据 1/5 的维度
                group_size = num_dim // 5 
                
                for group_name, g_idx in feature_groups.items():
                    # 计算要剔除的索引范围
                    start, end = g_idx * group_size, (g_idx + 1) * group_size
                    keep_indices = [idx for idx in range(num_dim) if idx < start or idx >= end]
                    
                    # 准备数据
                    X_c_sub = X_c[:, keep_indices]
                    X_b_sub = X_b[:, keep_indices]
                    
                    scaler = StandardScaler()
                    split = len(X_c_sub) // 2
                    ref_X = scaler.fit_transform(X_c_sub[:split])
                    test_X = scaler.transform(np.concatenate([X_c_sub[split:], X_b_sub]))
                    test_y = np.concatenate([np.zeros(len(X_c_sub)-split), np.ones(len(X_b_sub))])
                    
                    # 运行检测
                    nn = NearestNeighbors(n_neighbors=5, metric="manhattan", n_jobs=-1)
                    nn.fit(ref_X)
                    dist, _ = nn.kneighbors(test_X)
                    
                    auc = roc_auc_score(test_y, dist.mean(axis=1))
                    corrected_auc = max(auc, 1-auc)
                    
                    # 实时输出和保存
                    print(f"  [-] {layer_name} 丢弃 {group_name:<10} | AUC: {corrected_auc:.4f}")
                    with open(output_file, "a") as f_out:
                        f_out.write(f"{attack},{layer_name},{group_name},{corrected_auc:.4f}\n")
            except Exception as e:
                print(f"  ❌ 出错 {attack} {layer_name}: {e}")

print(f"\n🏆 全量实验完成！结果文件: {output_file}")