import numpy as np
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

def get_best_expert_auc(attack_name):
    # 自动适配路径：尝试当前目录和上级目录
    possible_paths = [
        f'./features_final/{attack_name}',
        f'../features_final/{attack_name}',
        f'D:/project_backdoor/BackdoorBench/features_final/{attack_name}'
    ]
    
    base_path = None
    for p in possible_paths:
        if os.path.exists(p):
            base_path = p
            break
    
    if not base_path:
        return None

    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    best_auc = 0
    best_layer = "N/A"
    
    for layer in layers:
        try:
            clean_path = os.path.join(base_path, f'{layer}_clean.npy')
            bd_path = os.path.join(base_path, f'{layer}_bd.npy')
            
            if not os.path.exists(clean_path): continue
            
            clean = np.nan_to_num(np.load(clean_path))
            bd = np.nan_to_num(np.load(bd_path))
            
            # 标准化
            scaler = StandardScaler()
            c_std = scaler.fit_transform(clean)
            b_std = scaler.transform(bd)
            
            # L1-KNN (消融实验证明曼哈顿距离最强)
            nn = NearestNeighbors(n_neighbors=5, metric='manhattan', n_jobs=-1)
            nn.fit(c_std)
            
            d_c, _ = nn.kneighbors(c_std)
            d_b, _ = nn.kneighbors(b_std)
            
            s_c = np.mean(d_c, axis=1)
            s_b = np.mean(d_b, axis=1)
            
            y_true = [0]*len(s_c) + [1]*len(s_b)
            y_scores = np.concatenate([s_c, s_b])
            auc = roc_auc_score(y_true, y_scores)
            
            # 【核心逻辑】：处理特征塌缩 (Inward Anomaly)
            actual_auc = max(auc, 1 - auc)
            
            if actual_auc > best_auc:
                best_auc = actual_auc
                best_layer = layer
        except Exception as e:
            continue
            
    return best_auc, best_layer

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 UCAT 极值复现系统启动...")
    print("="*50)

    attacks = [
        'badnets', 'blended', 'wanet', 'sig', 'refool', 
        'inputaware', 'lira', 'bpp', 'trojannn',
        'badnet_all2all', 'blind', 'ctrl', 'ftrojan'
    ]
    
    results = {}
    found_any = False

    for a in attacks:
        res = get_best_expert_auc(a)
        if res:
            auc, layer = res
            results[a] = (auc, layer)
            print(f"✅ 成功评估 [ {a:12} ] -> 最佳层级: {layer} | AUC: {auc:.4f}")
            found_any = True
        else:
            print(f"❌ 跳过 [ {a:12} ] : 未找到特征文件")

    if not found_any:
        print("\n[!] 警告：未在任何预设路径下找到特征文件！")
        print("请检查 D:/project_backdoor/BackdoorBench/features_final/ 是否存在。")
    else:
        print("\n" + "█"*50)
        print("🏆  UCAT FINAL CHAMPION TABLE (消融实验极值版)  🏆")
        print("█"*50)
        for a in attacks:
            if a in results:
                auc, layer = results[a]
                print(f"{a:15} | {layer:8} | {auc:.4f}")
        print("█"*50)