import torch
import torch.nn as nn
import numpy as np
import os
import sys
import glob
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torchvision

# ==========================================
# 1. 路径配置 (严格锁定 94.81% 模型)
# ==========================================
PROJECT_ROOT = r"D:\project_backdoor\BackdoorBench"
CKPT_PATH = r"D:\project_backdoor\BackdoorBench\checkpoint\ckpt_high_acc.pth"
FEATURES_DIR = r"D:\project_backdoor\BackdoorBench\features_final"

def get_94_acc_model():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    print(f"[*] 正在加载核心干净模型 (94.81%): {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location='cuda')
    model.load_state_dict(ckpt['net'])
    return model.eval().cuda()

# ==========================================
# 2. 权重感知提取器
# ==========================================
class WeightAwareUCAT:
    def __init__(self, model):
        self.model = model
        with torch.no_grad():
            self.channel_scores = torch.sum(torch.abs(self.model.fc.weight.data), dim=0).cuda()
        self.layer4_output = None
        self.model.layer4.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.layer4_output = output

    def compute_5dim(self, act):
        B, C, H, W = act.shape
        f_mean = torch.mean(act, dim=(2, 3))
        f_std = torch.std(act, dim=(2, 3))
        flat_x = act.view(B, C, -1)
        flat_x = flat_x - flat_x.min(dim=2, keepdim=True)[0] + 1e-8
        sorted_x, _ = torch.sort(flat_x, dim=2)
        n = H * W
        idx = torch.arange(1, n + 1, device=act.device).float()
        f_gini = (torch.sum((2 * idx - n - 1) * sorted_x, dim=2)) / (n * torch.sum(sorted_x, dim=2))
        grid_y, grid_x = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij')
        grid_x, grid_y = grid_x.to(act.device), grid_y.to(act.device)
        denom = torch.sum(act, dim=(2, 3)) + 1e-8
        f_cx = torch.sum(act * grid_x, dim=(2, 3)) / denom
        f_cy = torch.sum(act * grid_y, dim=(2, 3)) / denom
        return torch.stack([f_mean, f_std, f_gini, f_cx, f_cy], dim=2)

# ==========================================
# 3. 执行流程：物理扫图
# ==========================================
def run():
    model = get_94_acc_model()
    extractor = WeightAwareUCAT(model)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ])

    # 1. 提取基准 (Standard CIFAR-10)
    print("\n[*] 正在提取干净基准...")
    clean_testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
    clean_features = []
    for i in tqdm(range(min(2000, len(clean_testset))), desc="Clean Base"):
        img, _ = clean_testset[i]
        img_tensor = transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            model(img_tensor)
            raw = extractor.compute_5dim(extractor.layer4_output)
            clean_features.append((raw * extractor.channel_scores.view(1, 512, 1)).reshape(1, -1).cpu().numpy())
    feat_cl = np.concatenate(clean_features, axis=0)

    # 2. 物理扫图：直接去 record 文件夹搜图片
    tasks = ['sig_attack_1', 'ctrl_attack_final', 'ucat_official_test']
    for folder in tasks:
        search_path = os.path.join(PROJECT_ROOT, 'record', folder, '**', '*.png')
        print(f"\n🚀 正在物理搜索文件夹: {folder}")
        # 递归搜索所有子文件夹里的 png
        img_paths = glob.glob(search_path, recursive=True)
        # 排除掉那些可能存在的 clean 备份，只取 bd_test_dataset 里的
        img_paths = [p for p in img_paths if 'bd_test' in p or 'poison' in p]
        
        print(f"[*] 物理发现中毒图片数量: {len(img_paths)}")
        if not img_paths: continue

        bd_features = []
        for p in tqdm(img_paths[:2000], desc=f"Processing {folder}"): # 每种取2000张够用了
            img = Image.open(p).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).cuda()
            with torch.no_grad():
                model(img_tensor)
                raw = extractor.compute_5dim(extractor.layer4_output)
                bd_features.append((raw * extractor.channel_scores.view(1, 512, 1)).reshape(1, -1).cpu().numpy())

        if bd_features:
            feat_bd = np.concatenate(bd_features, axis=0)
            alias = folder.split('_')[0]
            save_folder = os.path.join(FEATURES_DIR, f"{alias}_weighted_clean")
            os.makedirs(save_folder, exist_ok=True)
            np.save(os.path.join(save_folder, 'layer4_bd.npy'), feat_bd)
            np.save(os.path.join(save_folder, 'layer4_clean.npy'), feat_cl)
            for l in ['layer1', 'layer2', 'layer3']:
                np.save(os.path.join(save_folder, f"{l}_bd.npy"), np.zeros((feat_bd.shape[0], 10)))
                np.save(os.path.join(save_folder, f"{l}_clean.npy"), np.zeros((feat_cl.shape[0], 10)))
            print(f"✅ {alias} 提取完成！")

if __name__ == "__main__":
    run()