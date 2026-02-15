import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免阻塞

def visualize_features_tsne():
    """
    加载特征文件，进行TSNE降维并可视化
    """
    
    # =====================
    # 加载特征文件
    # =====================
    poison_path = "features_badnet_poison.npy"
    clean_path = "features_badnet_clean.npy"

    if not os.path.exists(poison_path):
        print(f"错误: 文件不存在 {poison_path}")
        return
    if not os.path.exists(clean_path):
        print(f"错误: 文件不存在 {clean_path}")
        return

    print("加载特征文件...")
    poison = np.load(poison_path)
    clean = np.load(clean_path)
    print(f"后门特征形状: {poison.shape}")
    print(f"干净特征形状: {clean.shape}")

    # =====================
    # 合并数据和标签
    # =====================
    X = np.vstack([poison, clean])
    y = np.hstack([np.ones(len(poison)), np.zeros(len(clean))])
    
    print(f"\n合并后数据形状: {X.shape}")
    print(f"标签分布: 后门={np.sum(y==1)}, 干净={np.sum(y==0)}")
    
    # =====================
    # TSNE降维
    # =====================
    print("\n执行TSNE降维...")
    tsne = TSNE(n_components=2, 
            random_state=42, 
            perplexity=30,
            max_iter=1000,
            verbose=1)
    X_tsne = tsne.fit_transform(X)
    
    print(f"降维后形状: {X_tsne.shape}")
    
    # =====================
    # 可视化
    # =====================
    print("\n绘制散点图...")
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    # 绘制干净样本（蓝色）
    mask_clean = y == 0
    ax.scatter(X_tsne[mask_clean, 0], X_tsne[mask_clean, 1],
              c='blue', label='Clean', alpha=0.6, s=50, edgecolors='navy')
    
    # 绘制后门样本（红色）
    mask_poison = y == 1
    ax.scatter(X_tsne[mask_poison, 0], X_tsne[mask_poison, 1],
              c='red', label='Poisoned', alpha=0.6, s=50, edgecolors='darkred')
    
    # 标签和标题
    ax.set_xlabel('TSNE Component 1', fontsize=12, fontweight='bold')
    ax.set_ylabel('TSNE Component 2', fontsize=12, fontweight='bold')
    ax.set_title('TSNE Visualization of BadNet Features', fontsize=14, fontweight='bold')
    
    # 图例
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    
    # 网格线
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # =====================
    # 保存图片
    # =====================
    output_dir = "./figures/0214"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "tsne_badnet.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    print(f"\n✓ 图片已保存: {output_path}")
    
    # 显示图片（如果是在有图形界面的环境中，否则可以注释掉）
    # plt.show()
    
    # =====================
    # 输出统计信息
    # =====================
    print("\n" + "="*50)
    print("特征统计信息")
    print("="*50)
    print(f"\n后门样本 (n={len(poison)}):")
    print(f"  均值: {poison.mean(axis=0)}")
    print(f"  标准差: {poison.std(axis=0)}")
    
    print(f"\n干净样本 (n={len(clean)}):")
    print(f"  均值: {clean.mean(axis=0)}")
    print(f"  标准差: {clean.std(axis=0)}")
    
    # 计算特征差异
    mean_diff = np.abs(poison.mean(axis=0) - clean.mean(axis=0))
    print(f"\n特征均值差异: {mean_diff}")
    print(f"最大差异特征索引: {np.argmax(mean_diff)}")

if __name__ == "__main__":
    visualize_features_tsne()