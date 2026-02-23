import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

# 创建保存图片的文件夹
os.makedirs('figures/0217', exist_ok=True)

# 层列表
layers = ['layer1', 'layer2', 'layer3', 'layer4']
results = {}

# 为每一层加载数据并计算交叉验证 AUC
for layer in layers:
    # 加载特征
    poison = np.load(f'features_badnet_poison_{layer}.npy')
    clean = np.load(f'features_badnet_clean_{layer}.npy')
    X = np.vstack([poison, clean])
    y = np.hstack([np.ones(len(poison)), np.zeros(len(clean))])

    # 逻辑回归，5折交叉验证
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    
    results[layer] = {
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs)
    }
    print(f"{layer}: AUC = {results[layer]['mean_auc']:.4f} ± {results[layer]['std_auc']:.4f}")

# 绘制柱状图
layers_list = list(results.keys())
means = [results[l]['mean_auc'] for l in layers_list]
stds = [results[l]['std_auc'] for l in layers_list]

plt.figure(figsize=(8,5))
plt.bar(layers_list, means, yerr=stds, capsize=5, color='skyblue', edgecolor='navy')
plt.ylim(0.5, 1.0)
plt.ylabel('AUC')
plt.title('Layer-wise Feature Discriminability for BadNets')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('figures/0217/layer_auc_comparison.png', dpi=150)
plt.show()

# 绘制所有层的 ROC 曲线（使用固定划分）
plt.figure(figsize=(8,6))
for layer in layers:
    poison = np.load(f'features_wanet_poison_{layer}.npy')
    clean = np.load(f'features_wanet_clean_{layer}.npy')
    X = np.vstack([poison, clean])
    y = np.hstack([np.ones(len(poison)), np.zeros(len(clean))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{layer} (AUC={roc_auc:.3f})')

plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Layers')
plt.legend()
plt.savefig('figures/0222/layer_roc_comparison.png', dpi=150)
plt.show()
