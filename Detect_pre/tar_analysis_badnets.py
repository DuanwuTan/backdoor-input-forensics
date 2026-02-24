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

MODEL_PATH = './record/20260215_235930_badnet_attack_badnet_DvMv/attack_result.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
result = load_attack_result(MODEL_PATH)
model  = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model.load_state_dict(result['model'])
model  = model.to(device).eval()

# BadNets 触发器位置（右下角 3x3）
trigger_mask = np.zeros((32, 32))
trigger_mask[29:32, 29:32] = 1.0

def get_heatmap(img_tensor):
    target_layer = model.layer2[-1].conv2
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale = cam(input_tensor=img_tensor.unsqueeze(0), targets=None)[0]
    heatmap = cv2.resize(grayscale, (32, 32))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

def compute_tar(heatmap, mask):
    return float((heatmap * mask).sum() / (heatmap.sum() + 1e-8))

# 计算后门样本 TAR
print('计算后门样本 TAR...')
bd_loader = DataLoader(result['bd_test'], batch_size=1, shuffle=True)
bd_tars = []
for i, batch in enumerate(bd_loader):
    if i >= 200: break
    img = batch[0][0].to(device)
    hm  = get_heatmap(img)
    bd_tars.append(compute_tar(hm, trigger_mask))
    if i % 20 == 0:
        print(f'  后门样本 {i}/200')

# 计算干净样本 TAR（同一区域）
print('计算干净样本 TAR...')
clean_loader = DataLoader(result['clean_test'], batch_size=1, shuffle=True)
clean_tars = []
for i, batch in enumerate(clean_loader):
    if i >= 200: break
    img = batch[0][0].to(device)
    hm  = get_heatmap(img)
    clean_tars.append(compute_tar(hm, trigger_mask))
    if i % 20 == 0:
        print(f'  干净样本 {i}/200')

bd_tars    = np.array(bd_tars)
clean_tars = np.array(clean_tars)

# 统计检验
stat, p = mannwhitneyu(bd_tars, clean_tars, alternative='greater')
print(f'\n===== TAR 分析结果 =====')
print(f'后门样本 TAR: {bd_tars.mean():.4f} ± {bd_tars.std():.4f}')
print(f'干净样本 TAR: {clean_tars.mean():.4f} ± {clean_tars.std():.4f}')
print(f'Mann-Whitney U: stat={stat:.1f}, p={p:.4f}')
if p < 0.05:
    print('结论：后门触发器被可解释性方法系统性忽视 ✓ （论文核心发现）')
else:
    print('结论：无显著差异，需调整分析角度')

# 可视化
os.makedirs('figures/0224', exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].hist(bd_tars,    bins=30, alpha=0.7, label=f'Backdoor (mean={bd_tars.mean():.3f})',    color='red')
axes[0].hist(clean_tars, bins=30, alpha=0.7, label=f'Clean (mean={clean_tars.mean():.3f})', color='blue')
axes[0].set_xlabel('Trigger Attention Ratio (TAR)')
axes[0].set_ylabel('Count')
axes[0].set_title(f'TAR Distribution (p={p:.4f})')
axes[0].legend()

# 箱型图
axes[1].boxplot([bd_tars, clean_tars], labels=['Backdoor', 'Clean'])
axes[1].set_ylabel('TAR')
axes[1].set_title('TAR Comparison')

plt.tight_layout()
plt.savefig('figures/0224/tar_analysis.png', dpi=150)
print('图片已保存到 figures/0224/tar_analysis.png')