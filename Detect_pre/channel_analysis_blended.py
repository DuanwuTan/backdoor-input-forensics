import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os

layers = ['layer1', 'layer2', 'layer3', 'layer4']
top_k = 20

# 创建输出文件夹
os.makedirs('figures/0223/blended_channels', exist_ok=True)

important_channels = {}

for layer in layers:
    poison = np.load(f'features_blended_poison_{layer}.npy')
    clean = np.load(f'features_blended_clean_{layer}.npy')
    X = np.vstack([poison, clean])
    y = np.hstack([np.ones(len(poison)), np.zeros(len(clean))])
    
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(X, y)
    
    coef = clf.coef_[0]
    top_idx = np.argsort(np.abs(coef))[-top_k:][::-1]
    important_channels[layer] = top_idx
    
    print(f"{layer}: 重要通道索引 {top_idx}")
    
    # 绘制通道均值对比图
    poison_mean = poison.mean(axis=0)[top_idx]
    clean_mean = clean.mean(axis=0)[top_idx]
    
    plt.figure(figsize=(10, 4))
    x = np.arange(top_k)
    width = 0.35
    plt.bar(x - width/2, poison_mean, width, label='Poison', color='red', alpha=0.7)
    plt.bar(x + width/2, clean_mean, width, label='Clean', color='blue', alpha=0.7)
    plt.xlabel('Channel Rank')
    plt.ylabel('Mean Activation')
    plt.title(f'Blended {layer} Top {top_k} Important Channels')
    plt.legend()
    plt.savefig(f'figures/0223/blended_channels/{layer}_channels.png', dpi=150)
    plt.close()

# 分析跨层重叠
sets = {layer: set(important_channels[layer]) for layer in layers}
common_all = set.intersection(*sets.values())
print(f"\n所有层共有的重要通道索引: {common_all}")