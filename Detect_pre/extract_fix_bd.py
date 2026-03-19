import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import sys, os

# 1. 强制对齐项目路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.resnet import ResNet18
from Detect_pre.unified_extractor_final import UCAT_Extractor # 确保导入你的提取器

def fix_extraction():
    # 待修复的列表 (按照你昨晚的路径填)
    attacks = {
        "badnets": "./record/20260215_.../attack_result.pt",
        "wanet": "./record/20260221_.../attack_result.pt",
        "lira": "./record/lira_folder/attack_result.pt",
        "bpp": "./record/bpp_folder/attack_result.pt",
        # ... 其他攻击
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name, path in attacks.items():
        print(f"\n>>> 正在重新打磨 [ {name} ] 的毒药...")
        
        # 加载攻击包
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # A. 实例化模型 (使用昨晚成功的适配逻辑)
        model = ResNet18(num_classes=10).to(device)
        model.load_state_dict(checkpoint['model'])
        model.eval()

        # B. 获取“正宗”的后门数据集
        # 重点：不要手动加触发器，直接从 checkpoint 里拿测试集
        # BackdoorBench 的 attack_result 包含 bd_test 数据集对象
        bd_dataset = checkpoint['bd_test']
        
        # 如果 bd_test 是个复杂的包装类，我们这样解包
        bd_loader = DataLoader(bd_dataset, batch_size=64, shuffle=False)

        # C. 使用你的 UCAT 提取器
        extractor = UCAT_Extractor(model)
        all_features = {f"layer{i}": [] for i in [1, 2, 3, 4]}

        print(f"正在提取正宗后门特征...")
        count = 0
        for batch_x, batch_y, *rest in bd_loader:
            batch_x = batch_x.to(device)
            
            # 顺便验证一下 ASR (可选，但这能确保模型真的认出这些毒)
            # preds = model(batch_x).argmax(dim=1)
            # print(f"Batch ASR: {(preds == checkpoint['args'].target_label).float().mean()}")

            with torch.no_grad():
                features = extractor.extract(batch_x)
                for layer, feat in features.items():
                    all_features[layer].append(feat.cpu().numpy())
            
            count += 1
            if count >= 16: break # 提取 1024 张

        # D. 保存覆盖旧的错误特征
        save_dir = f"./features_final/{name}"
        os.makedirs(save_dir, exist_ok=True)
        for layer, data_list in all_features.items():
            final_feat = np.concatenate(data_list, axis=0)
            np.save(f"{save_dir}/{layer}_bd.npy", final_feat)
            print(f"✅ {layer}_bd.npy 已更新! 形状: {final_feat.shape}")

if __name__ == "__main__":
    fix_extraction()