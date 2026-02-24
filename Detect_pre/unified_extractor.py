import sys, os
sys.path.append('.')
import torch
import numpy as np
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.save_load_attack import load_attack_result

MODEL_PATHS = {
    'badnets': './record/20260215_235930_badnet_attack_badnet_DvMv/attack_result.pt',
    'blended': './record/20260221_010831_blended_attack_blended_Aniv/attack_result.pt',
    'wanet':   './record/20260221_013811_wanet_attack_wanet_CF3H/attack_result.pt',
    'sig':     './record/sig_attack_1/attack_result.pt',
    'refool':  './record/refool_attack_2/attack_result.pt',
    'inputaware': './record/inputaware_attack_1/attack_result.pt',
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']

def load_model(attack_name):
    result = load_attack_result(MODEL_PATHS[attack_name])
    model  = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(result['model'])
    return model.to(device).eval(), result

def extract_channel_means(model, loader, layer_name, max_samples=500):
    layer = dict(model.named_modules())[layer_name]
    feats = []
    count = 0
    def hook(m, i, o):
        feats.append(o.detach().cpu().mean(dim=[2,3]).numpy())
    handle = layer.register_forward_hook(hook)
    for batch in loader:
        if count >= max_samples: break
        with torch.no_grad():
            _ = model(batch[0].to(device))
        count += len(batch[0])
    handle.remove()
    return np.vstack(feats)

def run(attack_name):
    print(f'\n===== {attack_name.upper()} =====')
    model, result = load_model(attack_name)
    clean_loader = DataLoader(result['clean_test'], batch_size=64, shuffle=False)
    bd_loader    = DataLoader(result['bd_test'],    batch_size=64, shuffle=False)
    os.makedirs('features', exist_ok=True)
    for layer in LAYERS:
        cf = extract_channel_means(model, clean_loader, layer)
        bf = extract_channel_means(model, bd_loader, layer)
        np.save(f'features/{attack_name}_{layer}_clean.npy', cf)
        np.save(f'features/{attack_name}_{layer}_bd.npy', bf)
        print(f'  {layer}: clean={cf.shape}, bd={bf.shape}')

if __name__ == '__main__':
    for atk in ['badnets', 'blended', 'wanet']:
        run(atk)
    print('\n完成！等新模型训练好再跑后三个')