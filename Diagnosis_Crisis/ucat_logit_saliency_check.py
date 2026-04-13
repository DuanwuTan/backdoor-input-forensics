import torch
import torch.nn as nn
import numpy as np
import os
import sys
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import glob

# ==========================================
# 1. 配置 (使用架构一致的 7x7 模型)
# ==========================================
PROJECT_ROOT = r"D:\project_backdoor\BackdoorBench"
CLEAN_7x7 = r"D:\project_backdoor\BackdoorBench\record\resnet18_base_for_trojan\attack_result.pt"
POISON_7x7 = r"D:\project_backdoor\BackdoorBench\record\sig_attack_1\attack_result.pt"

def get_model_7x7(path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    ckpt = torch.load(path, map_location='cuda', weights_only=False)
    sd = ckpt['net'] if 'net' in ckpt else (ckpt['model'] if 'model' in ckpt else ckpt)
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    return model.eval().cuda()

# ==========================================
# 2. 核心诊断：Logit 显著性分析
# ==========================================
def analyze_logits():
    c_model = get_model_7x7(CLEAN_7x7)
    p_model = get_model_7x7(POISON_7x7)

    # 准备 SIG 中毒图像
    img_paths = glob.glob(os.path.join(PROJECT_ROOT, 'record', 'sig_attack_1', '**', '*.png'), recursive=True)
    img_paths = [p for p in img_paths if 'bd_test' in p][:100]
    
    trans = transforms.Compose([
        transforms.Resize((32,32)), transforms.ToTensor(), 
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ])

    c_logits_list, p_logits_list = [], []

    print("\n[*] 正在提取 Logits 决策信息...")
    for p in tqdm(img_paths):
        img = trans(Image.open(p).convert('RGB')).unsqueeze(0).cuda()
        with torch.no_grad():
            c_out = c_model(img) # [1, 10]
            p_out = p_model(img) # [1, 10]
            c_logits_list.append(c_out.cpu())
            p_logits_list.append(p_out.cpu())

    c_logits = torch.cat(c_logits_list, dim=0) # [100, 10]
    p_logits = torch.cat(p_logits_list, dim=0) # [100, 10]

    # --- 关键指标计算 ---
    def get_sharpness(logits):
        # 计算最高分和次高分的差距 (Margin)
        vals, _ = torch.topk(logits, 2, dim=1)
        margin = vals[:, 0] - vals[:, 1]
        # 计算 Logits 的标准差 (整体波动)
        std = torch.std(logits, dim=1)
        return margin.mean().item(), std.mean().item()

    c_margin, c_std = get_sharpness(c_logits)
    p_margin, p_std = get_sharpness(p_logits)

    print(f"\n📊 [决策层 (Logits) 显著性对比]")
    print(f"{'指标':<20} | {'干净模型 (误报现场)':<20} | {'中毒模型 (劫持现场)':<20}")
    print("-" * 65)
    print(f"{'Max-Next Margin':<20} | {c_margin:<20.4f} | {p_margin:<20.4f}")
    print(f"{'Logit Std':<20} | {c_std:<20.4f} | {p_std:<20.4f}")
    
    print(f"\n💡 深度反思:")
    print(f"   中毒模型的 Margin 通常是干净模型的数倍。")
    print(f"   如果 Gap 巨大，说明 UCAT 翻盘的终极算法是：")
    print(f"   特征异常得分 (Feature Score) × 决策显著性 (Logit Sharpness)")

if __name__ == "__main__":
    analyze_logits()