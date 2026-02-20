import sys
sys.path.append("D:\\project_backdoor\\BackdoorBench")
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from utils.save_load_attack import load_attack_result
import matplotlib.pyplot as plt
import os

os.makedirs('figures/0221', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./record/20260215_235930_badnet_attack_badnet_DvMv/attack_result.pt"
attack_result = load_attack_result(model_path)
model_state = attack_result['model']

# 重建模型（与训练一致，标准结构，不修改 conv1）
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)
model.load_state_dict(model_state)
model = model.to(device)
model.eval()

# 获取一个后门样本和一个干净样本
poison_dataset = attack_result['bd_test']
clean_dataset = attack_result['clean_test']
poison_img = poison_dataset[0][0].unsqueeze(0).to(device)
clean_img = clean_dataset[0][0].unsqueeze(0).to(device)

# 定义目标层：你可以尝试 layer3 或 layer4 的最后一个卷积层
# 打印模型结构来确认层名
print(model)
import matplotlib.pyplot as plt

# 将图像转换为 numpy 并显示
img_np = poison_img.squeeze().cpu().numpy().transpose(1, 2, 0)  # 转换为 HWC 格式
# 如果图像值在 [0,1] 之间，直接显示；如果是在 [-1,1] 之间，可能需要调整
plt.imshow(img_np)
plt.title('Poison Sample - First Image')
plt.axis('off')
plt.savefig('figures/0221/poison_sample.png', dpi=150)
plt.show()
# 通常 ResNet-18 的 layer4 最后一个卷积是 model.layer4[1].conv2
target_layers = [model.layer3[-1].conv2]  # 或者 model.layer3[-1].conv2

# 创建 GradCAM 对象
cam = GradCAM(model=model, target_layers=target_layers)

# 对后门样本计算热力图
grayscale_cam = cam(input_tensor=poison_img, targets=None)
grayscale_cam = grayscale_cam[0, :]

# 将热力图叠加到原图
img = poison_img.squeeze().cpu().numpy().transpose(1,2,0)
# 归一化到 [0,1] 用于显示
img = (img - img.min()) / (img.max() - img.min() + 1e-8)
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

plt.imshow(cam_image)
plt.title(f'Grad-CAM on {target_layers[0]}')
plt.axis('off')
plt.savefig('figures/0221/gradcam_layer4.png', dpi=150)
plt.show()
print(f"热力图数值范围: min={grayscale_cam.min():.4f}, max={grayscale_cam.max():.4f}")
# with torch.no_grad():
#     out = model.layer3(poison_img)
# print("layer3 output shape:", out.shape)   # 应该输出 (1, 256, 2, 2)
# ===== 第一步：确认触发器的真实位置 =====
import matplotlib.patches as patches

# 获取样本图像
img_np = poison_img.squeeze().cpu().numpy().transpose(1,2,0)
# 归一化到 [0,1] 用于显示（如果之前已经归一化过，可能不需要）
img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

# 显示图像并画出右下角可能的触发器区域（假设 4x4 方块）
fig, ax = plt.subplots(1)
ax.imshow(img_np)
# CIFAR-10 图像是 32x32，右下角坐标约为 (28,28) 到 (32,32)
rect = patches.Rectangle((28, 28), 4, 4, linewidth=2, edgecolor='green', facecolor='none')
ax.add_patch(rect)
plt.title('Poison Sample with Trigger Region (右下角)')
plt.savefig('figures/0221/poison_with_trigger_box.png')
plt.show()

print("已保存 poison_with_trigger_box.png，请查看右下角绿色框内是否有白色方块。")
# ===== 第二步：检查预测类别和热力图数值分布 =====
with torch.no_grad():
    logits = model(poison_img)
    pred_class = logits.argmax(dim=1).item()
    prob = torch.softmax(logits, dim=1)[0, pred_class].item()
true_label = poison_dataset[0][1]  # 真实标签
print(f"真实标签: {true_label}")
print(f"预测类别: {pred_class}, 置信度: {prob:.4f}")

# 查看热力图的最大值位置
import numpy as np
h, w = grayscale_cam.shape
max_idx = np.argmax(grayscale_cam)
max_y, max_x = np.unravel_index(max_idx, grayscale_cam.shape)
print(f"热力图最大值位置: (y={max_y}, x={max_x})  对应原图大致区域")

# 将热力图缩放到原图尺寸并保存带坐标的图
cam_resized = cv2.resize(grayscale_cam, (32, 32))
plt.figure()
plt.imshow(img_np)
plt.imshow(cam_resized, cmap='jet', alpha=0.5)
plt.scatter([max_x * 32 / w], [max_y * 32 / h], color='white', marker='x', s=100)  # 标注最大值点
plt.title(f'Grad-CAM with max point (pred={pred_class})')
plt.savefig('figures/0221/gradcam_with_maxpoint.png')
plt.show()
from pytorch_grad_cam import AblationCAM

cam_ablation = AblationCAM(model=model, target_layers=target_layers)
grayscale_cam_ab = cam_ablation(input_tensor=poison_img, targets=None)[0, :]
# 缩放并显示
cam_ab_resized = cv2.resize(grayscale_cam_ab, (32, 32))
plt.imshow(img_np)
plt.imshow(cam_ab_resized, cmap='jet', alpha=0.5)
plt.title('Ablation-CAM')
plt.savefig('figures/0221/ablation_cam.png')
plt.show()

# 打印最大值位置
h_ab, w_ab = grayscale_cam_ab.shape
max_idx_ab = np.argmax(grayscale_cam_ab)
max_y_ab, max_x_ab = np.unravel_index(max_idx_ab, grayscale_cam_ab.shape)
print(f"Ablation-CAM 最大值位置: (y={max_y_ab}, x={max_x_ab})")