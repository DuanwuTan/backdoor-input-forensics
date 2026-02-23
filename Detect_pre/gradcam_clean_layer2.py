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
os.makedirs('figures/0222/clean_layer2', exist_ok=True)

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

clean_dataset = attack_result['clean_test']

# 目标层：layer2 的最后一个卷积层
target_layers = [model.layer2[-1].conv2]
cam = GradCAM(model=model, target_layers=target_layers)

num_samples = 20  # 取前20个干净样本
results = []

for idx in range(num_samples):
    img, label = clean_dataset[idx]  # 返回 (img, label)
    img_tensor = img.unsqueeze(0).to(device)

    grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0, :]
    h, w = grayscale_cam.shape  # 应为 4x4

    max_idx = np.argmax(grayscale_cam)
    max_y, max_x = np.unravel_index(max_idx, grayscale_cam.shape)
    results.append((max_y, max_x, label))
    print(f"干净样本 {idx} (标签={label}): 最大值位置 ({max_y}, {max_x})")

    # 保存热力图（前10个）
    if idx < 10:
        cam_resized = cv2.resize(grayscale_cam, (32, 32))
        img_np = img.numpy().transpose(1, 2, 0)  # (32,32,3)
        # 归一化到 [0,1] 显示
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        plt.imshow(img_np)
        plt.imshow(cam_resized, cmap='jet', alpha=0.5)
        plt.scatter(max_x * 32 / w, max_y * 32 / h, color='white', marker='x', s=50)
        plt.title(f'Clean sample {idx} (label={label})')
        plt.savefig(f'figures/0222/clean_layer2/clean_{idx}_gradcam.png')
        plt.close()

# 统计各位置出现次数
unique, counts = np.unique([(r[0], r[1]) for r in results], axis=0, return_counts=True)
print("\n=== 干净样本 layer2 热力图最大值分布 ===")
for pos, cnt in zip(unique, counts):
    print(f"位置 {pos}: {cnt} 次")

# 可选：绘制直方图
pos_labels = [f"({y},{x})" for y,x in zip([r[0] for r in results], [r[1] for r in results])]
plt.figure()
plt.hist(pos_labels)
plt.title('Distribution of max positions on clean samples (layer2)')
plt.xlabel('Position (row, col)')
plt.ylabel('Count')
plt.savefig('figures/0222/clean_layer2/position_hist.png')
plt.show()