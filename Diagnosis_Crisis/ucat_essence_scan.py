import torch
import torch.nn as nn
import numpy as np
import os
import glob
import pandas as pd
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
from torch.utils.data import DataLoader

# ==========================================
# 1. 架构与路径配置
# ==========================================
PROJECT_ROOT = r"D:\project_backdoor\BackdoorBench"
CLEAN_7x7 = r"D:\project_backdoor\BackdoorBench\record\resnet18_base_for_trojan\attack_result.pt"
SIG_7x7 = r"D:\project_backdoor\BackdoorBench\record\sig_attack_1\attack_result.pt"

def load_model(path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    ckpt = torch.load(path, map_location='cuda', weights_only=False)
    sd = ckpt['net'] if 'net' in ckpt else (ckpt['model'] if 'model' in ckpt else ckpt)
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    return model.eval().cuda()

# ==========================================
# 2. 核心度量函数
# ==========================================
def get_metrics(model, img_tensors):
    acts = []
    # Hook Layer4 输出
    def hook(m, i, o): acts.append(torch.mean(o, dim=(2,3))) # GAP 后的特征 [B, 512]
    handle = model.layer4.register_forward_hook(hook)
    
    with torch.no_grad():
        logits = model(img_tensors)
    handle.remove()
    
    # 获取当前模型面对这一组样本的平均激活特征 [512]
    mean_act = torch.stack(acts).mean(0).mean(0) 
    
    # A. 强度 (Intensity): 特征偏离的绝对物理强度 (L1 范数)
    intensity = torch.norm(mean_act, p=1).item()
    
    # B. 对齐度 (Alignment): 活跃通道与决策权重的重合度
    # 物理激活 Top-20
    _, top_phys_idx = torch.topk(mean_act, 20)
    phys_set = set(top_phys_idx.cpu().numpy())
    
    # 决策权重显著性 Top-20 (Std 越大说明该通道越具有决策指向性)
    weight_spec = torch.std(model.fc.weight.data, dim=0)
    _, top_spec_idx = torch.topk(weight_spec, 20)
    spec_set = set(top_spec_idx.cpu().numpy())
    
    alignment = len(phys_set & spec_set)
    
    # C. 决策边际 (Margin): 最高分与次高分的差距，衡量决策的偏执程度
    vals, _ = torch.topk(logits, 2, dim=1)
    margin = (vals[:, 0] - vals[:, 1]).mean().item()
    
    return intensity, alignment, margin

# ==========================================
# 3. 四象限全量扫描
# ==========================================
def run_full_scan():
    print("[*] 正在加载模型和数据...")
    m_clean = load_model(CLEAN_7x7)
    m_poison = load_model(SIG_7x7)
    
    trans = transforms.Compose([
        transforms.Resize((32,32)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ])

    # --- 获取中毒样本 (X_poison) ---
    img_paths_bd = glob.glob(os.path.join(PROJECT_ROOT, 'record', 'sig_attack_1', '**', '*.png'), recursive=True)
    img_paths_bd = [p for p in img_paths_bd if 'bd_test' in p][:100]
    t_bd = torch.stack([trans(Image.open(p).convert('RGB')) for p in img_paths_bd]).cuda()

    # --- 获取干净样本 (X_clean) ---
    # 修正：直接加载标准 CIFAR-10 测试集
    print("[*] 正在加载标准 CIFAR-10 作为干净样本...")
    clean_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=trans)
    clean_loader = DataLoader(clean_dataset, batch_size=100, shuffle=False)
    t_cl, _ = next(iter(clean_loader))
    t_cl = t_cl.cuda()

    quadrants = [
        ("1. M_clean + X_clean (Normal)", m_clean, t_cl),
        ("2. M_clean + X_poison (Misreport)", m_clean, t_bd),
        ("3. M_poison + X_clean (Hidden)", m_poison, t_cl),
        ("4. M_poison + X_poison (Attack)", m_poison, t_bd)
    ]

    results = []
    for name, m, t in quadrants:
        inten, align, marg = get_metrics(m, t)
        results.append({
            "Quadrant": name, 
            "Intensity (物理响应)": round(inten, 2), 
            "Alignment (决策对齐)": align, 
            "Margin (决策边际)": round(marg, 2)
        })
    
    df = pd.DataFrame(results)
    print("\n" + "="*95)
    print("🏆 UCAT 后门行为学：四象限深度解析报告")
    print("="*95)
    print(df.to_string(index=False))
    print("="*95)
    
    print("\n💡 关键数据解读：")
    print("1. 对比 2 和 4：物理强度可能差不多，但第 4 象限的 Margin（13+）应远高于第 2 象限（3-4）。")
    print("2. 对比 1 和 2：物理强度会暴涨，这就是为什么 UCAT 1.0 会在第 2 象限产生误报。")
    print("3. 后门的本质：是物理响应强度(Intensity)与决策劫持(Margin)的【高维共振】。")

if __name__ == "__main__":
    run_full_scan()