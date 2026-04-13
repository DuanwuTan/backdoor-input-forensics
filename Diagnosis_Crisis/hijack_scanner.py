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
# 0. 解决 PyTorch 2.6 加载安全限制
# ==========================================
import torch.serialization
torch.serialization.add_safe_globals([np._core.multiarray.scalar, np._core.multiarray._reconstruct])

# ==========================================
# 1. 配置与架构 (锁定 94.81% 干净 vs SIG 中毒)
# ==========================================
PROJECT_ROOT = r"D:\project_backdoor\BackdoorBench"
CLEAN_CKPT = r"D:\project_backdoor\BackdoorBench\checkpoint\ckpt_high_acc.pth"
SIG_CKPT = r"D:\project_backdoor\BackdoorBench\record\sig_attack_1\attack_result.pt"

def load_model(path):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    
    # 核心修复点：添加 weights_only=False
    ckpt = torch.load(path, map_location='cuda', weights_only=False)
    
    sd = ckpt['net'] if 'net' in ckpt else (ckpt['model'] if 'model' in ckpt else ckpt)
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    return model.eval().cuda()

# ==========================================
# 2. 核心诊断逻辑：Top-K 权重-激活对齐分析
# ==========================================
def scan_hijack(attack_name, poisoned_ckpt):
    print(f"\n" + "="*60)
    print(f"🔎 正在扫描后门劫持特征: {attack_name}")
    print("="*60)

    # A. 加载模型
    clean_model = load_model(CLEAN_CKPT)
    poison_model = load_model(poisoned_ckpt)

    # B. 提取 FC 权重显著性 (使用 Std 衡量专一性)
    # 计算 512 个通道在 10 个类别权重的标准差
    c_weight_specificity = torch.std(clean_model.fc.weight.data, dim=0).cuda()
    p_weight_specificity = torch.std(poison_model.fc.weight.data, dim=0).cuda()

    # C. 获取该攻击的中毒图像 (物理扫描)
    img_paths = glob.glob(os.path.join(PROJECT_ROOT, 'record', f'{attack_name}_attack_1', '**', '*.png'), recursive=True)
    img_paths = [p for p in img_paths if 'bd_test' in p][:100]
    
    if not img_paths:
        print("❌ 未发现中毒图片，请检查路径。")
        return

    transform = transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ])

    c_temp, p_temp = [], []

    # 注册 Hook 提取 Layer4 物理激活
    def get_act(storage):
        def hook(m, i, o): storage.append(torch.mean(o, dim=(0, 2, 3)))
        return hook
    
    h1 = clean_model.layer4.register_forward_hook(get_act(c_temp))
    h2 = poison_model.layer4.register_forward_hook(get_act(p_temp))

    for p in tqdm(img_paths, desc="提取物理信号"):
        img = transform(Image.open(p).convert('RGB')).unsqueeze(0).cuda()
        with torch.no_grad():
            clean_model(img)
            poison_model(img)
    
    # 计算通道平均物理激活强度 [512]
    c_phys_activation = torch.stack(c_temp).mean(dim=0)
    p_phys_activation = torch.stack(p_temp).mean(dim=0)

    # D. 对齐度分析 (Top-K 重合率)
    K = 50
    def analyze_alignment(phys, spec):
        # 物理激活最强的 Top-K 通道索引
        _, top_phys_idx = torch.topk(phys, K)
        top_phys_set = set(top_phys_idx.cpu().numpy())
        
        # 权重最显著的 Top-K 通道索引
        _, top_spec_idx = torch.topk(spec, K)
        top_spec_set = set(top_spec_idx.cpu().numpy())
        
        overlap = top_phys_set.intersection(top_spec_set)
        return len(overlap) / K, overlap

    c_rate, _ = analyze_alignment(c_phys_activation, c_weight_specificity)
    p_rate, p_overlap = analyze_alignment(p_phys_activation, p_weight_specificity)

    # E. 输出诊断结论
    print(f"\n📊 [诊断结果 - Top-{K} 决策对齐分析]")
    print(f"   - 干净模型 (Clean)  对齐率: {c_rate*100:.1f}%")
    print(f"   - 中毒模型 (Poison) 对齐率: {p_rate*100:.1f}%")
    print(f"   - 决策对齐增益 (Gap): {p_rate / (c_rate + 1e-8):.2f} 倍")

    print(f"\n💡 结论:")
    if p_rate > c_rate * 1.5:
        print(f"   ✅ 证据发现：中毒模型中，触发器激活的通道与高权重决策通道高度对齐。")
        print(f"   🚀 翻盘点：目前的 UCAT 只是做了乘法，没做 '对齐筛选'。我们需要一个 Top-K 对齐掩码。")
    else:
        print(f"   ⚠️ 对齐差异不明显，可能劫持发生在更高维的交互或空间分布中。")

if __name__ == "__main__":
    scan_hijack('sig', SIG_CKPT)