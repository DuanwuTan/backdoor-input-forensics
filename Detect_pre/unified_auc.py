import sys, os
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

attacks = ['badnets', 'blended', 'wanet']
layers  = ['layer1', 'layer2', 'layer3', 'layer4']

auc_matrix = np.zeros((len(attacks), len(layers)))

for i, atk in enumerate(attacks):
    for j, layer in enumerate(layers):
        cf = np.load(f'features/{atk}_{layer}_clean.npy')
        bf = np.load(f'features/{atk}_{layer}_bd.npy')
        n  = min(len(cf), len(bf))
        X  = np.vstack([cf[:n], bf[:n]])
        y  = np.hstack([np.zeros(n), np.ones(n)])
        clf = LogisticRegression(max_iter=1000)
        auc = cross_val_score(clf, X, y, cv=5, scoring='roc_auc').mean()
        auc_matrix[i, j] = auc
        print(f'{atk:10s} {layer}: AUC={auc:.4f}')

os.makedirs('figures/0224', exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(auc_matrix, cmap='RdYlGn', vmin=0.5, vmax=1.0)
ax.set_xticks(range(4)); ax.set_xticklabels(layers)
ax.set_yticks(range(len(attacks))); ax.set_yticklabels([a.upper() for a in attacks])
plt.colorbar(im)
for i in range(len(attacks)):
    for j in range(4):
        ax.text(j, i, f'{auc_matrix[i,j]:.3f}', ha='center', va='center', fontsize=11)
ax.set_title('AUC Heatmap: 3 Attacks x 4 Layers')
plt.tight_layout()
plt.savefig('figures/0224/auc_heatmap_3attacks.png', dpi=150)
print('\nAUC热力图已保存到 figures/0224/auc_heatmap_3attacks.png')