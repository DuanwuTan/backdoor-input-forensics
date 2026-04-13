import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# ==========================================
# 1. 路径配置
# ==========================================
FEATURES_DIR = r"D:\project_backdoor\BackdoorBench\features_final"

def evaluate_ucat(folder_name):
    base_path = os.path.join(FEATURES_DIR, folder_name)
    if not os.path.exists(base_path):
        return None

    # 我们重点关注 layer4，因为那是权重感知的核心层
    layers = ['layer4', 'layer3', 'layer2', 'layer1']
    results = {}

    for layer in layers:
        clean_path = os.path.join(base_path, f'{layer}_clean.npy')
        bd_path = os.path.join(base_path, f'{layer}_bd.npy')
        
        if not os.path.exists(clean_path) or not os.path.exists(bd_path):
            continue
            
        try:
            # 加载特征
            clean = np.load(clean_path)
            bd = np.load(bd_path)
            
            # 如果特征全是 0 (占位符)，跳过
            if np.all(clean == 0): continue
            
            # 1. 标准化 (KNN 对量纲敏感)
            scaler = StandardScaler()
            c_std = scaler.fit_transform(clean)
            b_std = scaler.transform(bd)
            
            # 2. L1-KNN (曼哈顿距离) - UCAT 的核心算法
            nn = NearestNeighbors(n_neighbors=5, metric='manhattan', n_jobs=-1)
            nn.fit(c_std)
            
            # 计算异常得分 (到 5 个最近邻干净样本的平均距离)
            d_c, _ = nn.kneighbors(c_std)
            d_b, _ = nn.kneighbors(b_std)
            
            scores_clean = np.mean(d_c, axis=1)
            scores_bd = np.mean(d_b, axis=1)
            
            # 3. 计算 AUC
            y_true = [0]*len(scores_clean) + [1]*len(scores_bd)
            y_scores = np.concatenate([scores_clean, scores_bd])
            auc = roc_auc_score(y_true, y_scores)
            
            # 处理特征空间内聚 (Inward Anomaly) 逻辑
            final_auc = max(auc, 1 - auc)
            results[layer] = final_auc
            
        except Exception as e:
            print(f"      [!] {layer} 计算出错: {e}")
            
    return results

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏆 UCAT 权重感知实验结果评测 (平反验证版)")
    print("="*60)

    # 自动扫描所有以 _weighted_clean 结尾的实验文件夹
    all_folders = [f for f in os.listdir(FEATURES_DIR) if os.path.isdir(os.path.join(FEATURES_DIR, f))]
    target_folders = [f for f in all_folders if '_weighted_clean' in f]

    if not target_folders:
        print("❌ 未发现 _weighted_clean 特征文件夹，请先运行特征提取脚本。")
    else:
        print(f"发现 {len(target_folders)} 个加权实验文件夹。正在计算...\n")
        
        summary = []
        for folder in target_folders:
            res = evaluate_ucat(folder)
            if res and 'layer4' in res:
                auc = res['layer4']
                status = "✅ 误报已平反" if auc < 0.7 else "⚠️ 仍有误报"
                if auc < 0.6: status = "🔥 完美平反 (接近随机)"
                
                summary.append((folder, auc, status))
                print(f"📂 {folder:30} | Layer4 AUC: {auc:.4f} | {status}")

        print("\n" + "█"*60)
        print("📊  FINAL RECAP: WEIGHT-AWARE DEFENSE PERFORMANCE")
        print("█"*60)
        print(f"{'Attack (Clean Model)':35} | {'Layer4 AUC':10}")
        print("-" * 60)
        for folder, auc, status in summary:
            print(f"{folder:35} | {auc:.4f}")
        print("█"*60)
        print("\n[注] 如果 AUC 接近 0.5，说明权重感知成功让干净模型不再对触发器产生异常反应。")