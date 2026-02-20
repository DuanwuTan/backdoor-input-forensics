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
print("脚本开始执行")
os.makedirs('figures/0221', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./record/20260215_235930_badnet_attack_badnet_DvMv/attack_result.pt"
attack_result = load_attack_result(model_path)
model_state = attack_result['model']

# 重建模型
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model.load_state_dict(model_state)
model = model.to(device)
model.eval()

poison_dataset = attack_result['bd_test']
target_layers = [model.layer3[-1].conv2]  # layer3 最后一个卷积
cam = GradCAM(model=model, target_layers=target_layers)

num_samples = 20
print(f"将处理 {num_samples} 个样本")
results = []

for idx in range(num_samples):
    img = poison_dataset[idx][0].unsqueeze(0).to(device)
    grayscale_cam = cam(input_tensor=img, targets=None)[0, :]
    h, w = grayscale_cam.shape
    max_idx = np.argmax(grayscale_cam)
    max_y, max_x = np.unravel_index(max_idx, grayscale_cam.shape)
    results.append((max_y, max_x))
    print(f"样本 {idx}: 最大值位置 ({max_y}, {max_x})")

    # 保存前5个样本的热力图
    if idx < 5:
        cam_resized = cv2.resize(grayscale_cam, (32, 32))
        img_np = img.squeeze().cpu().numpy().transpose(1,2,0)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        plt.imshow(img_np)
        plt.imshow(cam_resized, cmap='jet', alpha=0.5)
        plt.scatter(max_x * 32 / w, max_y * 32 / h, color='white', marker='x')
        plt.title(f'Sample {idx}')
        plt.savefig(f'figures/0221/multi_sample_{idx}.png')
        plt.close()

# 统计
unique, counts = np.unique(results, axis=0, return_counts=True)
print("\n各位置出现次数：")
for pos, cnt in zip(unique, counts):
    print(f"位置 {pos}: {cnt} 次")