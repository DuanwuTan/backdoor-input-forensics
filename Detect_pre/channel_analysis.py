import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import os

layers = ['layer1', 'layer2', 'layer3', 'layer4']
top_k = 20  # 取前 K 个最重要通道

# 存储每层的重要通道索引
important_channels = {}

for layer in layers:
    # 加载特征
    poison = np.load(f'features_badnet_poison_{layer}.npy')
    clean = np.load(f'features_badnet_clean_{layer}.npy')
    X = np.vstack([poison, clean])
    y = np.hstack([np.ones(len(poison)), np.zeros(len(clean))])
    
    # 训练逻辑回归（使用全部数据，不划分）
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(X, y)
    
    # 获取系数（权重），形状 (n_features,)
    coef = clf.coef_[0]
    
    # 取绝对值最大的 top_k 个通道的索引
    top_idx = np.argsort(np.abs(coef))[-top_k:][::-1]  # 从大到小排序
    important_channels[layer] = top_idx
    
    print(f"{layer}: 重要通道索引 {top_idx}")
    
    # 可选：绘制这些通道在两类样本上的均值对比
    poison_mean = poison.mean(axis=0)[top_idx]
    clean_mean = clean.mean(axis=0)[top_idx]
    
    plt.figure(figsize=(10, 4))
    x = np.arange(top_k)
    width = 0.35
    plt.bar(x - width/2, poison_mean, width, label='Poison', color='red', alpha=0.7)
    plt.bar(x + width/2, clean_mean, width, label='Clean', color='blue', alpha=0.7)
    plt.xlabel('Channel Rank')
    plt.ylabel('Mean Activation')
    plt.title(f'{layer} Top {top_k} Important Channels')
    plt.legend()
    plt.savefig(f'figures/0220/channels_{layer}.png', dpi=150)
    plt.show()

# 分析跨层重叠
# 将每层的重要通道索引转换为集合
sets = {layer: set(important_channels[layer]) for layer in layers}

# 计算所有层的交集
common_all = set.intersection(*sets.values())
print(f"\n所有层共有的重要通道索引: {common_all}")

# 可视化重叠（Venn 图需要安装 matplotlib_venn，可选）
try:
    from matplotlib_venn import venn3, venn2
    if len(layers) == 4:
        # 这里简单起见，只画前两层的重叠示例
        plt.figure()
        venn2([sets['layer1'], sets['layer2']], ('layer1', 'layer2'))
        plt.title('Overlap of Important Channels (layer1 vs layer2)')
        plt.savefig('figures/0220/venn_layer1_layer2.png', dpi=150)
        plt.show()
except ImportError:
    print("若要画 Venn 图，请安装 matplotlib_venn：pip install matplotlib-venn")