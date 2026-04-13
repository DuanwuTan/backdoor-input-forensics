import torch
import numpy as np
import os
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import glob

PROJECT_ROOT = r"D:\project_backdoor\BackdoorBench"
CLEAN_3x3 = r"D:\project_backdoor\BackdoorBench\checkpoint\ckpt_high_acc.pth"
CLEAN_7x7 = r"D:\project_backdoor\BackdoorBench\record\resnet18_base_for_trojan\attack_result.pt"
SIG_7x7 = r"D:\project_backdoor\BackdoorBench\record\sig_attack_1\attack_result.pt"

def get_model(path, k_size):
    model = models.resnet18(weights=None)
    if k_size == 3:
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.Identity()
    model.fc = torch.nn.Linear(512, 10)
    ckpt = torch.load(path, map_location='cuda', weights_only=False)
    sd = ckpt['net'] if 'net' in ckpt else (ckpt['model'] if 'model' in ckpt else ckpt)
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=False)
    return model.eval().cuda()

# 诊断函数
def diag_auc(clean_path, poison_path, c_size, p_size):
    print(f"\n对比: 干净({c_size}x{c_size}) vs 中毒({p_size}x{p_size})")
    c_model = get_model(clean_path, c_size)
    # 提取 Layer4 均值作为最简特征
    features = []
    def hook(m, i, o): features.append(torch.mean(o, dim=(0,2,3)).cpu().numpy())
    c_model.layer4.register_forward_hook(hook)

    # 加载 SIG 中毒图片
    img_paths = glob.glob(os.path.join(PROJECT_ROOT, 'record', 'sig_attack_1', '**', '*.png'), recursive=True)
    img_paths = [p for p in img_paths if 'bd_test' in p][:100]
    
    trans = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), 
                                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])

    for p in img_paths:
        img = trans(Image.open(p).convert('RGB')).unsqueeze(0).cuda()
        with torch.no_grad(): c_model(img)
    
    # 计算特征偏移强度 (简单模长)
    intensity = np.mean([np.linalg.norm(f) for f in features])
    print(f"   -> 响应强度: {intensity:.4f}")

# 执行诊断
diag_auc(CLEAN_3x3, SIG_7x7, 3, 7) # 危机现场
diag_auc(CLEAN_7x7, SIG_7x7, 7, 7) # 正统战场