import sys, os
sys.path.append('.')
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from utils.save_load_attack import load_attack_result

MODEL_PATH = './record/sig_attack_1/attack_result.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

result = load_attack_result(MODEL_PATH)
model  = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model.load_state_dict(result['model'])
model  = model.to(device).eval()

def get_heatmap(img_tensor):
    target_layer = model.layer2[-1].conv2
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale = cam(input_tensor=img_tensor.unsqueeze(0), targets=None)[0]
    heatmap = cv2.resize(grayscale, (32, 32))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

def compute_entropy(heatmap):
    h = heatmap.flatten()
    h = h / (h.sum() + 1e-8)
    entropy = -np.sum(h * np.log(h + 1e-8))
    return float(entropy)

print('计算后门样本熵...')
bd_loader = DataLoader(result['bd_test'], batch_size=1, shuffle=True)
bd_entropies = []
for i, batch in enumerate(bd_loader):
    if i >= 200: break
    img = batch[0][0].to(device)
    hm  = get_heatmap(img)
    bd_entropies.append(compute_entropy(hm))
    if i % 20 == 0:
        print(f'  后门样本 {i}/200')

print('计算干净样本熵...')
clean_loader = DataLoader(result['clean_test'], batch_size=1, shuffle=True)
clean_entropies = []
for i, batch in enumerate(clean_loader):
    if i >= 200: break
    img = batch[0][0].to(device)
    hm  = get_heatmap(img)
    clean_entropies.append(compute_entropy(hm))
    if i % 20 == 0:
        print(f'  干净样本 {i}/200')

bd_entropies    = np.array(bd_entropies)
clean_entropies = np.array(clean_entropies)

stat, p = mannwhitneyu(bd_entropies, clean_entropies, alternative='two-sided')
print(f'\n===== TAR 分析结果 (SIG) =====')
print(f'后门样本 熵: {bd_entropies.mean():.4f} ± {bd_entropies.std():.4f}')
print(f'干净样本 熵: {clean_entropies.mean():.4f} ± {clean_entropies.std():.4f}')
print(f'Mann-Whitney U: stat={stat:.1f}, p={p:.4f}')
if p < 0.05:
    print('结论：SIG 后门样本注意力分布与干净样本存在显著差异 ✓')
else:
    print('结论：无显著差异')

os.makedirs('figures/0225', exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(bd_entropies,    bins=30, alpha=0.7, label=f'Backdoor (mean={bd_entropies.mean():.3f})', color='red')
axes[0].hist(clean_entropies, bins=30, alpha=0.7, label=f'Clean (mean={clean_entropies.mean():.3f})', color='blue')
axes[0].set_xlabel('Heatmap Entropy')
axes[0].set_title(f'SIG TAR (p={p:.4f})')
axes[0].legend()
axes[1].boxplot([bd_entropies, clean_entropies], tick_labels=['Backdoor', 'Clean'])
axes[1].set_ylabel('Entropy')
axes[1].set_title('SIG Entropy Comparison')
plt.tight_layout()
plt.savefig('figures/0225/tar_analysis_sig.png', dpi=150)
print('图片已保存到 figures/0225/tar_analysis_sig.png')
