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

# ==========================================
# 0. 环境与路径配置
# ==========================================
PROJECT_ROOT = r"D:\project_backdoor\BackdoorBench"
CLEAN_7x7_PATH = r"D:\project_backdoor\BackdoorBench\record\resnet18_base_for_trojan\attack_result.pt"
CLEAN_3x3_PATH = r"D:\project_backdoor\BackdoorBench\checkpoint\ckpt_high_acc.pth"

# 12 种核心攻击文件夹映射 (排除 BPP)
ATTACK_LIST = {
    "BadNets":      "20260215_235930_badnet_attack_badnet_DvMv",
    "Blended":      "20260221_010831_blended_attack_blended_Aniv",
    "WaNet":        "20260221_013811_wanet_attack_wanet_CF3H",
    "SIG":          "sig_attack_1",
    "Refool":       "refool_attack_2",
    "InputAware":   "inputaware_attack_1",
    "LIRA":         "lira_attack_1",
    "FTrojan":      "ftrojan_attack_final",
    "TrojanNN":     "trojannn_attack_final",
    "Blind":        "blind_attack_final",
    "CTRL":         "ctrl_attack_final",
    "BadNet_A2A":   "badnet_all2all_final"
}

# ==========================================
# 1. 架构自适应加载
# ==========================================
def load_model_adaptive(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt['net'] if 'net' in ckpt else (ckpt['model'] if 'model' in ckpt else ckpt)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    
    # 检测卷积核大小
    k_size = sd['conv1.weight'].shape[2] 
    model = models.resnet18(weights=None)
    if k_size == 3:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    model.fc = nn.Linear(sd['fc.weight'].shape[1], sd['fc.weight'].shape[0])
    model.load_state_dict(sd, strict=False)
    return model.eval().cuda()

# ==========================================
# 2. 核心度量计算
# ==========================================
def get_metrics(model, img_tensors):
    acts = []
    def hook(m, i, o): acts.append(torch.mean(o, dim=(2,3))) 
    handle = model.layer4.register_forward_hook(hook)
    with torch.no_grad():
        logits = model(img_tensors)
    handle.remove()
    
    mean_act = torch.stack(acts).mean(0).mean(0) 
    intensity = torch.norm(mean_act, p=1).item()
    
    # 对齐度: Top-20 重合数
    _, top_p_idx = torch.topk(mean_act, 20)
    _, top_s_idx = torch.topk(torch.std(model.fc.weight.data, dim=0), 20)
    align = len(set(top_p_idx.cpu().numpy()) & set(top_s_idx.cpu().numpy()))
    
    # 决策边际
    v, _ = torch.topk(logits, 2, dim=1)
    margin = (v[:, 0] - v[:, 1]).mean().item()
    
    return intensity, align, margin

# ==========================================
# 3. 执行扫描
# ==========================================
def main():
    trans = transforms.Compose([
        transforms.Resize((32,32)), transforms.ToTensor(), 
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ])
    
    # 干净测试集
    clean_ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=trans)
    t_cl = torch.stack([clean_ds[i][0] for i in range(100)]).cuda()
    
    results = []
    for name, folder in ATTACK_LIST.items():
        print(f"\n🔍 正在扫描: {name}")
        path = os.path.join(PROJECT_ROOT, 'record', folder, 'attack_result.pt')
        if not os.path.exists(path):
            print(f"   ⚠️ 跳过，未找到文件")
            continue

        try:
            m_poison = load_model_adaptive(path)
            is_3x3 = (m_poison.conv1.kernel_size[0] == 3)
            m_clean = load_model_adaptive(CLEAN_3x3_PATH if is_3x3 else CLEAN_7x7_PATH)

            # 物理捞图
            imgs = glob.glob(os.path.join(PROJECT_ROOT, 'record', folder, '**', '*.png'), recursive=True)
            imgs = [p for p in imgs if 'bd_test' in p or 'poison' in p][:100]
            if not imgs: 
                print(f"   ⚠️ 没搜到 png 图片")
                continue
            t_bd = torch.stack([trans(Image.open(p).convert('RGB')) for p in imgs]).cuda()

            # 4 象限分析
            quads = [("M_cl+X_cl", m_clean, t_cl), ("M_cl+X_bd", m_clean, t_bd), 
                     ("M_po+X_cl", m_poison, t_cl), ("M_po+X_bd", m_poison, t_bd)]

            for q_n, m, t in quads:
                inten, align, marg = get_metrics(m, t)
                results.append({"Attack": name, "Quadrant": q_n, "Int": inten, "Align": align, "Margin": marg})
                if q_n in ["M_cl+X_bd", "M_po+X_bd"]:
                    print(f"   {q_n:12} | Int: {inten:7.2f} | Align: {align:2} | Margin: {marg:6.2f}")

        except Exception as e:
            print(f"   ❌ 出错: {e}")

    df = pd.DataFrame(results)
    df.to_csv("ucat_12_attacks_essence_report.csv", index=False)
    print("\n" + "="*80)
    print("✅ 12 种攻击全量扫描完成！数据已存至: ucat_12_attacks_essence_report.csv")
    print("="*80)

if __name__ == "__main__":
    main()