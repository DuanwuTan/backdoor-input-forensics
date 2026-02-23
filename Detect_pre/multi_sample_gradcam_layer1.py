import sys
sys.path.append("D:\\project_backdoor\\BackdoorBench")
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from utils.save_load_attack import load_attack_result
import os

# 创建输出目录
os.makedirs('figures/0222/layer1/poison', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# BadNets 模型路径
model_path = "./record/20260215_235930_badnet_attack_badnet_DvMv/attack_result.pt"
attack_result = load_attack_result(model_path)
model_state = attack_result['model']

# 重建模型
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model.load_state_dict(model_state)
model = model.to(device)
model.eval()

# 后门数据集
poison_dataset = attack_result['bd_test']

# 目标层：layer1 的最后一个卷积层（输出 8×8）
target_layers = [model.layer1[-1].conv2]
cam = GradCAM(model=model, target_layers=target_layers)

num_samples = 20
results = []

for idx in range(num_samples):
    # 数据集返回格式可能是 (img, label) 或 (img, label, index)，统一用第一个元素作为图像
    sample = poison_dataset[idx]
    img = sample[0].unsqueeze(0).to(device)
    label = sample[1]  # 真实标签

    grayscale_cam = cam(input_tensor=img, targets=None)[0, :]  # shape (8,8)
    h, w = grayscale_cam.shape
    max_idx = np.argmax(grayscale_cam)
    max_y, max_x = np.unravel_index(max_idx, grayscale_cam.shape)
    results.append((max_y, max_x, label))
    print(f"后门样本 {idx} (真实标签={label}): 最大值位置 ({max_y}, {max_x})")

    # 保存前10个样本的热力图
    if idx < 10:
        cam_resized = cv2.resize(grayscale_cam, (32, 32))
        img_np = sample[0].numpy().transpose(1, 2, 0)  # (32,32,3)
        # 归一化到 [0,1] 显示
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        plt.figure()
        plt.imshow(img_np)
        plt.imshow(cam_resized, cmap='jet', alpha=0.5)
        plt.scatter(max_x * 32 / w, max_y * 32 / h, color='white', marker='x', s=50)
        plt.title(f'Poison sample {idx} (label={label})')
        plt.savefig(f'figures/0222/layer1/poison/poison_{idx}_gradcam.png')
        plt.close()

# 统计最大值位置分布
unique, counts = np.unique([(r[0], r[1]) for r in results], axis=0, return_counts=True)
print("\n=== 后门样本 layer1 热力图最大值分布 ===")
for pos, cnt in zip(unique, counts):
    print(f"位置 {pos}: {cnt} 次")

# 可选：保存坐标数组供后续分析
np.save('figures/0222/layer1/poison_pos.npy', np.array([(r[0], r[1]) for r in results]))