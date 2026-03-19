import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import importlib.util

# 🚀 1. 物理锁定根目录
root_path = os.getcwd() 
if root_path not in sys.path:
    sys.path.insert(0, root_path)
print(f"当前项目根目录已锁定: {root_path}")

# 🚀 2. 物理加载 models/resnet.py (绕过 ModuleNotFoundError)
resnet_path = os.path.join(root_path, 'models', 'resnet.py')
if not os.path.exists(resnet_path):
    # 尝试备用路径 (如果你的目录结构稍微不同)
    resnet_path = os.path.join(root_path, 'resnet.py')

print(f"正在物理加载模型文件: {resnet_path}")
spec = importlib.util.spec_from_file_location("resnet_module", resnet_path)
resnet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resnet_module)
# 现在我们可以拿到 ResNet18 了
ResNet18 = resnet_module.ResNet18 

# 🚀 3. 同理加载提取器 (如果提取器也在 Detect_pre 下)
from Detect_pre.unified_extractor_final import UCAT_Extractor 

print("✅ 模型与提取器物理加载成功！")

# ... 后面接你的 rescue() 函数 ...
def rescue():
    attacks = {
        "badnets": "./record/20260215_.../attack_result.pt",
        "wanet": "./record/20260221_.../attack_result.pt",
        "inputaware": "./record/inputaware_attack_1/attack_result.pt",
        "lira": "./record/lira_folder/attack_result.pt",
        "bpp": "./record/bpp_folder/attack_result.pt"
    }
    
    device = torch.device("cuda")
    for name, path in attacks.items():
        print(f"正在拯救 {name} 的正宗特征...")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        
        # 实例化模型
        model = ResNet18(num_classes=10).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()

        # 【核心操作】直接读官方后门数据集，不准手动加触发器！
        bd_loader = DataLoader(ckpt['bd_test'], batch_size=64, shuffle=False)
        
        extractor = UCAT_Extractor(model)
        feats = {f"layer{i}": [] for i in [1,2,3,4]}
        
        for i, (batch_x, batch_y, *rest) in enumerate(bd_loader):
            if i >= 16: break # 1024张
            with torch.no_grad():
                f_dict = extractor.extract(batch_x.to(device))
                for l, f in f_dict.items():
                    feats[l].append(f.cpu().numpy())
        
        # 覆盖保存
        for l, data in feats.items():
            np.save(f"./features_final/{name}/{l}_bd.npy", np.concatenate(data))
        print(f"✅ {name} 拯救成功！")

if __name__ == "__main__": rescue()