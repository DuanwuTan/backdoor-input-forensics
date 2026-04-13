import os
import sys
import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# 强行引入框架路径
PROJECT_ROOT = r"D:\project_backdoor\BackdoorBench"
sys.path.append(PROJECT_ROOT)
from utils.save_load_attack import load_attack_result

def get_high_acc_model():
    import torchvision.models as models
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model

def compute_features(feat_map):
    B, C, H, W = feat_map.shape
    flat = feat_map.view(B, C, -1)
    mean_val = flat.mean(dim=2)
    std_val = flat.std(dim=2) + 1e-6
    max_val, _ = flat.max(dim=2)
    papr = max_val / (mean_val + 1e-8)
    sorted_flat, _ = torch.sort(flat, dim=2)
    n = sorted_flat.size(2)
    indices = torch.arange(1, n+1, device=flat.device).float()
    gini = (2 * torch.sum(sorted_flat * indices, dim=2) - (n+1) * torch.sum(sorted_flat, dim=2)) / (n * torch.sum(sorted_flat, dim=2) + 1e-8)
    y_indices = torch.arange(H, device=feat_map.device).view(1, 1, H, 1).expand(B, C, H, W).float()
    x_indices = torch.arange(W, device=feat_map.device).view(1, 1, 1, W).expand(B, C, H, W).float()
    sum_act = feat_map.sum(dim=(2, 3)) + 1e-8
    centroid_y = (feat_map * y_indices).sum(dim=(2, 3)) / sum_act
    centroid_x = (feat_map * x_indices).sum(dim=(2, 3)) / sum_act
    return torch.stack([mean_val, gini, std_val, centroid_x, centroid_y, papr], dim=-1).view(B, -1).cpu().numpy()

def run_fpr_test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载模型
    model = get_high_acc_model().to(device)
    try:
        checkpoint = torch.load('./checkpoint/ckpt_high_acc.pth')
        model.load_state_dict(checkpoint['net'])
    except Exception as e:
        print(f"无法加载干净模型权重: {e}")
        return
    model.eval()

    # 2. 准备标准干净测试集
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    clean_test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    clean_loader = DataLoader(clean_test_set, batch_size=128, shuffle=False)

    # 3. 遍历 record 目录
    record_root = r'D:\project_backdoor\BackdoorBench\record'
    attacks = [d for d in os.listdir(record_root) if os.path.isdir(os.path.join(record_root, d))]
    
    # 注册 Hook
    features_dict = {f'layer{i}': [] for i in range(1, 5)}
    def get_hook(name):
        def hook(m, i, o): features_dict[name].append(compute_features(o))
        return hook
    hooks = [getattr(model, f'layer{i}').register_forward_hook(get_hook(f'layer{i}')) for i in range(1, 5)]

    print(f"\n{'Attack Name':<35} | {'ASR':<8} | {'Best AUC (FPR)':<15}")
    print("-" * 70)

    for attack in attacks:
        attack_path = os.path.join(record_root, attack, 'attack_result.pt')
        if not os.path.exists(attack_path):
            continue
        
        # 清空特征缓存
        for k in features_dict: features_dict[k] = []
        
        try:
            # 使用官方函数进行恢复，避开底层序列化错误
            data = load_attack_result(attack_path)
            
            # 自动探测中毒数据集
            bd_ds = data.get('bd_test') or data.get('test_bd') or data.get('bd_test_loader')
            if bd_ds is None:
                print(f"跳过 {attack}: 找不到中毒数据集键值")
                continue
            
            # 如果是 DataLoader 则提取 Dataset
            if isinstance(bd_ds, DataLoader):
                bd_ds = bd_ds.dataset
            
            # 过滤掉坏掉的字符串格式
            if isinstance(bd_ds, str):
                print(f"跳过 {attack}: bd_test 仅为字符串标签")
                continue

            bd_loader = DataLoader(bd_ds, batch_size=128, shuffle=False)
            
            labels = []
            bd_to_target = 0
            total_bd = 0
            
            with torch.no_grad():
                # 提取干净基准
                for i, t in clean_loader:
                    model(i.to(device))
                    labels.extend([0] * i.size(0))
                
                # 提取中毒激活
                for batch in bd_loader:
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    bd_to_target += predicted.eq(targets).sum().item()
                    total_bd += targets.size(0)
                    labels.extend([1] * inputs.size(0))
            
            asr = bd_to_target / total_bd
            
            layer_aucs = []
            for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
                feats = np.concatenate(features_dict[layer], axis=0)
                train_feats = feats[np.array(labels) == 0]
                scaler = StandardScaler().fit(train_feats)
                train_scaled = scaler.transform(train_feats)
                test_scaled = scaler.transform(feats)
                knn = NearestNeighbors(n_neighbors=5, metric='manhattan').fit(train_scaled)
                dist, _ = knn.kneighbors(test_scaled)
                auc = roc_auc_score(labels, dist.mean(axis=1))
                layer_aucs.append(max(auc, 1-auc))
            
            print(f"{attack[:35]:<35} | {asr:.4f} | {max(layer_aucs):.4f}")

        except Exception as e:
            print(f"解析攻击 {attack} 时出错: {str(e)[:50]}")
            continue

    for h in hooks: h.remove()

if __name__ == '__main__':
    run_fpr_test()