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

os.makedirs('figures/0222/poison_layer2', exist_ok=True)

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
target_layers = [model.layer2[-1].conv2]
cam = GradCAM(model=model, target_layers=target_layers)

num_samples = 20
results = []

for idx in range(num_samples):
    img, label = poison_dataset[idx][0], poison_dataset[idx][1]  # 注意解包方式
    img_tensor = img.unsqueeze(0).to(device)
    grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0, :]  # shape (32,32)
    h, w = grayscale_cam.shape
    max_idx = np.argmax(grayscale_cam)
    max_y, max_x = np.unravel_index(max_idx, grayscale_cam.shape)
    results.append((max_y, max_x, label))
    print(f"后门样本 {idx} (目标标签={label}): 最大值位置 ({max_y}, {max_x})")
    
    # 保存前10个样本的图
    if idx < 10:
        img_np = img.numpy().transpose(1,2,0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        plt.imshow(img_np)
        plt.imshow(grayscale_cam, cmap='jet', alpha=0.5)
        plt.scatter(max_x, max_y, color='white', marker='x', s=50)
        plt.title(f'Poison sample {idx} (target={label})')
        plt.savefig(f'figures/0222/poison_layer2/poison_{idx}_gradcam.png')
        plt.close()

# 统计
unique, counts = np.unique([(r[0], r[1]) for r in results], axis=0, return_counts=True)
print("\n=== 后门样本 layer2 热力图最大值分布 ===")
for pos, cnt in zip(unique, counts):
    print(f"位置 {pos}: {cnt} 次")