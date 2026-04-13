import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import glob

# ==========================================
# 1. 配置 (架构一致的 7x7 对抗)
# ==========================================
PROJECT_ROOT = r"D:\project_backdoor\BackdoorBench"
CLEAN_7x7 = r"D:\project_backdoor\BackdoorBench\record\resnet18_base_for_trojan\attack_result.pt"
POISON_7x7 = r"D:\project_backdoor\BackdoorBench\record\sig_attack_1\attack_result.pt"

def get_model_7x7(path):
    model = models.resnet18(weights=None) # 7x7 是默认架构
    model.fc = nn.Linear(512, 10)
    ckpt = torch.load(path, map_location='cuda', weights_only=False)
    sd = ckpt['net'] if 'net' in ckpt else (ckpt['model'] if 'model' in ckpt else ckpt)
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    return model.eval().cuda()

# ==========================================
# 2. 核心诊断：对齐度 (Alignment) 检查
# ==========================================
def check_alignment():
    c_model = get_model_7x7(CLEAN_7x7)
    p_model = get_model_7x7(POISON_7x7)

    # 获取 FC 层权重显著性 (Std 衡量劫持专一性)
    c_weights_spec = torch.std(c_model.fc.weight.data, dim=0) # [512]
    p_weights_spec = torch.std(p_model.fc.weight.data, dim=0) # [512]

    # 提取物理激活
    img_paths = glob.glob(os.path.join(PROJECT_ROOT, 'record', 'sig_attack_1', '**', '*.png'), recursive=True)
    img_paths = [p for p in img_paths if 'bd_test' in p][:100]
    trans = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), 
                                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])

    c_acts, p_acts = [], []
    def hook_c(m, i, o): c_acts.append(torch.mean(o, dim=(0,2,3)))
    def hook_p(m, i, o): p_acts.append(torch.mean(o, dim=(0,2,3)))
    c_model.layer4.register_forward_hook(hook_c)
    p_model.layer4.register_forward_hook(hook_p)

    for p in tqdm(img_paths, desc="提取物理激活"):
        img = trans(Image.open(p).convert('RGB')).unsqueeze(0).cuda()
        with torch.no_grad():
            c_model(img)
            p_model(img)

    c_phys = torch.stack(c_acts).mean(dim=0) # 干净模型面对 SIG 的通道激活
    p_phys = torch.stack(p_acts).mean(dim=0) # 中毒模型面对 SIG 的通道激活

    # 计算 Top-20 对齐度
    def get_alignment(phys, spec):
        _, top_p = torch.topk(phys, 20)
        _, top_s = torch.topk(spec, 20)
        overlap = len(set(top_p.cpu().numpy()) & set(top_s.cpu().numpy()))
        return overlap

    c_overlap = get_alignment(c_phys, c_weights_spec)
    p_overlap = get_alignment(p_phys, p_weights_spec)

    print(f"\n📊 [Top-20 权重-激活对齐度检查]")
    print(f"   - 干净模型重合数量: {c_overlap} / 20")
    print(f"   - 中毒模型重合数量: {p_overlap} / 20")
    print(f"   - 结论: {'劫持证据发现' if p_overlap > c_overlap else '需要进一步深挖'}")

if __name__ == "__main__":
    check_alignment()