import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch.utils.data import DataLoader

# ==========================================
# 架构适配：标准 Torchvision ResNet-18 (7x7 conv1)
# 针对 badnet_all2all_final 的权重结构定制
# ==========================================
def get_standard_resnet18(num_classes=10):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ==========================================
# UCAT 特征计算逻辑
# ==========================================
def compute_features(feat_map):
    B, C, H, W = feat_map.shape
    flat = feat_map.view(B, C, -1)
    mean_val = flat.mean(dim=2)
    std_val = flat.std(dim=2)
    max_val, _ = flat.max(dim=2)
    papr = max_val / (mean_val + 1e-8)
    
    sorted_flat, _ = torch.sort(flat, dim=2)
    n = sorted_flat.size(2)
    indices = torch.arange(1, n+1, device=flat.device).float()
    gini = (2 * torch.sum(sorted_flat * indices, dim=2) - (n+1) * torch.sum(sorted_flat, dim=2)) / (n * torch.sum(sorted_flat, dim=2) + 1e-8)
    gini[torch.isnan(gini)] = 0
    
    y_indices = torch.arange(H, device=feat_map.device).view(1, 1, H, 1).expand(B, C, H, W).float()
    x_indices = torch.arange(W, device=feat_map.device).view(1, 1, 1, W).expand(B, C, H, W).float()
    sum_act = feat_map.sum(dim=(2, 3)) + 1e-8
    centroid_y = (feat_map * y_indices).sum(dim=(2, 3)) / sum_act
    centroid_x = (feat_map * x_indices).sum(dim=(2, 3)) / sum_act
    features = torch.stack([mean_val, gini, std_val, centroid_x, centroid_y, papr], dim=-1)
    return features.view(B, -1).cpu().numpy()

def evaluate_ucat(model, clean_loader, bd_loader, device):
    print("\n[状态] 正在执行特征提取过程...")
    features_dict = {'layer1': [], 'layer2': [], 'layer3': [], 'layer4': []}
    
    def get_hook(layer_name):
        def hook(module, input, output):
            features_dict[layer_name].append(compute_features(output))
        return hook

    hooks = [
        model.layer1.register_forward_hook(get_hook('layer1')),
        model.layer2.register_forward_hook(get_hook('layer2')),
        model.layer3.register_forward_hook(get_hook('layer3')),
        model.layer4.register_forward_hook(get_hook('layer4'))
    ]

    model.eval()
    all_clean_correct = 0
    all_bd_correct = 0
    clean_total = 0
    bd_total = 0
    labels = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(clean_loader, desc="[基准] 干净样本"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_clean_correct += predicted.eq(targets).sum().item()
            clean_total += targets.size(0)
            labels.extend([0] * inputs.size(0))
            
        for batch in tqdm(bd_loader, desc="[目标] 中毒样本"):
            # 中毒数据集包装类可能返回多个值，取前两个
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            # ASR 计算：被分类为错误标签（劫持）的比例
            all_bd_correct += predicted.eq(targets).sum().item()
            bd_total += targets.size(0)
            labels.extend([1] * inputs.size(0))

    for h in hooks: h.remove()
    labels = np.array(labels)

    print(f"\n[验证] 准确率 (ACC): {all_clean_correct/clean_total*100:.2f}%")
    print(f"[验证] 攻击成功率 (ASR): {all_bd_correct/bd_total*100:.2f}%")

    print("\n[决策] 各层级 UCAT 检测效能 (AUC):")
    for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
        feats = np.concatenate(features_dict[layer], axis=0)
        train_feats = feats[labels == 0]
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_feats)
        test_scaled = scaler.transform(feats)
        knn = NearestNeighbors(n_neighbors=5, metric='manhattan')
        knn.fit(train_scaled)
        distances, _ = knn.kneighbors(test_scaled)
        scores = distances.mean(axis=1)
        auc = roc_auc_score(labels, scores)
        print(f"Layer {layer[-1]} AUC: {auc:.4f}")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    attack_path = r'D:\project_backdoor\BackdoorBench\record\badnet_all2all_final\attack_result.pt'
    
    if not os.path.exists(attack_path):
        print(f"致命错误: 无法定位攻击记录文件 {attack_path}")
        sys.exit(1)

    # 1. 强制外部加载干净数据，规避字符串解析错误
    print("[1/4] 正在下载/加载外部标准测试集...")
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    clean_test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    clean_loader = DataLoader(clean_test_set, batch_size=128, shuffle=False)

    # 2. 解析中毒记录
    print(f"[2/4] 正在解析中毒模型记录...")
    data = torch.load(attack_path, map_location='cpu', weights_only=False)
    
    # 3. 初始化并加载模型
    print("[3/4] 正在对齐架构与权重...")
    num_classes = data.get('num_classes', 10)
    model = get_standard_resnet18(num_classes=num_classes).to(device)
    
    state_dict = data['model']
    # 处理可能的 state_dict 嵌套
    if hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"架构对齐失败: {e}")
        sys.exit(1)

    # 4. 提取中毒数据集
    print("[4/4] 正在提取触发器数据集...")
    bd_ds = data['bd_test']
    bd_loader = DataLoader(bd_ds, batch_size=128, shuffle=False)

    # 5. 执行评估
    evaluate_ucat(model, clean_loader, bd_loader, device)