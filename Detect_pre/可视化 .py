import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 读取数据
df = pd.read_csv("UCAT_Full_Group_Ablation.csv")

# 2. 我们选一个最具代表性的层级，比如 Layer 2，来看看不同特征的重要性
# 或者是做一个全量汇总
for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
    pivot_df = df[df['Layer'] == layer].pivot(index="Attack", columns="Dropped_Group", values="AUC")
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title(f"Ablation Analysis: AUC when a feature group is dropped ({layer})")
    plt.savefig(f"ablation_heatmap_{layer}.png")
    print(f"✅ 已生成 {layer} 的消融热力图")