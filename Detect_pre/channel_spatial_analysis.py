import sys
sys.path.append("D:\\project_backdoor\\BackdoorBench")
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.save_load_attack import load_attack_result
import os

os.makedirs('figures/0221', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./record/20260215_235930_badnet_attack_badnet_DvMv/attack_result.pt"
attack_result = load_attack_result(model_path)
model_state = attack_result['model']

model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model.load_state_dict(model_state)
model = model.to(device)
model.eval()

poison_dataset = attack_result['bd_test']
img = poison_dataset[0][0].unsqueeze(0).to(device)

# 注册 hook 获取 layer3 输出
activations = []
def hook_fn(module, input, output):
    activations.append(output.detach().cpu())
handle = model.layer3.register_forward_hook(hook_fn)

with torch.no_grad():
    _ = model(img)
layer3_out = activations[0].squeeze().numpy()  # (256, 2, 2)
handle.remove()

# 重要通道索引
important_idx = np.array([248, 105, 135, 223, 7, 147, 224, 73, 214, 102, 12, 62, 245, 200, 222, 82, 91, 125, 215, 29])
# 提取这些通道的 2x2 激活图
important_maps = layer3_out[important_idx]  # (20, 2, 2)

# 显示前几个通道的空间分布
for i, ch in enumerate(important_idx[:5]):
    map_2x2 = important_maps[i]
    map_32x32 = cv2.resize(map_2x2, (32, 32), interpolation=cv2.INTER_LINEAR)
    plt.imshow(map_32x32, cmap='hot')
    plt.title(f'Channel {ch} activation (up sampled)')
    plt.colorbar()
    plt.savefig(f'figures/0221/channel_{ch}_spatial.png')
    plt.show()

# 计算每个通道的峰值位置（在 2x2 中）
peak_positions = []
for i in range(len(important_idx)):
    map_2x2 = important_maps[i]
    max_idx = np.argmax(map_2x2)
    pos = np.unravel_index(max_idx, (2,2))
    peak_positions.append(pos)
    print(f"通道 {important_idx[i]} 峰值位置: {pos}")

# 统计峰值位置分布
unique, counts = np.unique(peak_positions, axis=0, return_counts=True)
print("\n重要通道峰值位置分布：")
for pos, cnt in zip(unique, counts):
    print(f"位置 {pos}: {cnt} 个通道")