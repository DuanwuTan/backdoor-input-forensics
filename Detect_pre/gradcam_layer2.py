import sys
sys.path.append("D:\\project_backdoor\\BackdoorBench")
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
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
poison_img = poison_dataset[0][0].unsqueeze(0).to(device)

# 使用 layer2 的最后一个卷积层
target_layers = [model.layer2[-1].conv2]
cam = GradCAM(model=model, target_layers=target_layers)

grayscale_cam = cam(input_tensor=poison_img, targets=None)[0, :]
print(f"layer2 热力图形状: {grayscale_cam.shape}")  # 应为 (4,4)
h, w = grayscale_cam.shape
max_idx = np.argmax(grayscale_cam)
max_y, max_x = np.unravel_index(max_idx, grayscale_cam.shape)
print(f"最大值位置 (原始热力图坐标): ({max_y}, {max_x})")
print(f"对应原图大致区域: 根据 4x4 划分")

# 叠加显示
cam_resized = cv2.resize(grayscale_cam, (32, 32))
img_np = poison_img.squeeze().cpu().numpy().transpose(1,2,0)
img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
overlay = show_cam_on_image(img_np, cam_resized, use_rgb=True)
plt.imshow(overlay)
plt.title('Grad-CAM on layer2')
plt.savefig('figures/0221/gradcam_layer2.png')
plt.show()