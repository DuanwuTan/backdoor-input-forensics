import sys, os
sys.path.append('.')
import torch
import numpy as np
from torchvision import models
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from utils.save_load_attack import load_attack_result
from sklearn.decomposition import PCA

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_PATHS = {
    'badnets': './record/20260215_235930_badnet_attack_badnet_DvMv/attack_result.pt',
    'blended': './record/20260221_010831_blended_attack_blended_Aniv/attack_result.pt',
    'wanet':   './record/20260221_013811_wanet_attack_wanet_CF3H/attack_result.pt',
    'sig':     './record/sig_attack_1/attack_result.pt',
    'refool':  './record/refool_attack_2/attack_result.pt',
    'inputaware': './record/inputaware_attack_1/attack_result.pt',
}

LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']

def load_model(attack_name):
    result = load_attack_result(MODEL_PATHS[attack_name])
    model  = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 10)
    model.load_state_dict(result['model'])
    return model.to(device).eval(), result

def extract_channel_means_batch(model, loader, max_samples=1000):
    all_feats = {ln: [] for ln in LAYERS}
    layers_dict = dict(model.named_modules())
    handles = []

    for ln in LAYERS:
        def make_hook(name):
            def hook(m, i, o):
                all_feats[name].append(
                    o.detach().cpu().mean(dim=[2,3]).numpy())
            return hook
        handles.append(layers_dict[ln].register_forward_hook(make_hook(ln)))

    count = 0
    for batch in loader:
        if count >= max_samples: break
        with torch.no_grad():
            _ = model(batch[0].to(device))
        count += len(batch[0])

    for h in handles: h.remove()
    return {ln: np.vstack(all_feats[ln]) for ln in LAYERS}
class UCATDetector:
    def __init__(self, model, clean_loader):
        self.model    = model
        self.detectors = {}
        self.pcas      = {}
        print('  建立干净基线...')
        feats = extract_channel_means_batch(model, clean_loader, max_samples=1000)
        for ln in LAYERS:
            f = feats[ln]
            from sklearn.decomposition import PCA
            from sklearn.ensemble import IsolationForest
            n_components = min(30, f.shape[0]-1, f.shape[1])
            pca = PCA(n_components=n_components)
            f_reduced = pca.fit_transform(f)
            clf = IsolationForest(contamination=0.05, random_state=0, n_estimators=100)
            clf.fit(f_reduced)
            self.pcas[ln]      = pca
            self.detectors[ln] = clf
            print(f'    {ln}: 降维 {f.shape[1]}→{n_components}，基线建立完成')

    def score_batch(self, loader, max_samples=500):
        feats = extract_channel_means_batch(self.model, loader, max_samples)
        layer_scores = []
        for ln in LAYERS:
            f_reduced = self.pcas[ln].transform(feats[ln])
            # decision_function 越负越异常，取负号使越高越异常
            scores = -self.detectors[ln].decision_function(f_reduced)
            layer_scores.append(scores)
        return np.max(np.stack(layer_scores), axis=0)
def evaluate(attack_name):
    print(f'\n===== {attack_name.upper()} =====')
    model, result = load_model(attack_name)
    clean_loader  = DataLoader(result['clean_test'], batch_size=64, shuffle=True)
    bd_loader     = DataLoader(result['bd_test'],    batch_size=64, shuffle=True)

    detector      = UCATDetector(model, clean_loader)
    clean_scores  = detector.score_batch(clean_loader, max_samples=500)
    bd_scores     = detector.score_batch(bd_loader,    max_samples=500)

    y_true  = np.hstack([np.zeros(len(clean_scores)), np.ones(len(bd_scores))])
    y_score = np.hstack([clean_scores, bd_scores])
    auroc   = roc_auc_score(y_true, y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    fpr95 = fpr[np.argmin(np.abs(tpr - 0.95))]

    print(f'  AUROC:     {auroc:.4f}')
    print(f'  FPR@95TPR: {fpr95:.4f}')
    return {'attack': attack_name, 'auroc': auroc, 'fpr95': fpr95}

if __name__ == '__main__':
    results = []
    for atk in ['badnets', 'blended', 'wanet', 'sig', 'refool', 'inputaware']:
        r = evaluate(atk)
        results.append(r)

    print('\n' + '='*45)
    print(f'{"攻击":<12} {"AUROC":>8} {"FPR@95TPR":>12}')
    print('-'*45)
    for r in results:
        print(f'{r["attack"]:<12} {r["auroc"]:>8.4f} {r["fpr95"]:>12.4f}')

    os.makedirs('figures/0225', exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    attacks = [r['attack'] for r in results]
    aurocs  = [r['auroc']  for r in results]
    bars = ax.bar(attacks, aurocs, color=['#e74c3c','#3498db','#2ecc71'])
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel('AUROC')
    ax.set_title('UCAT Detection Performance')
    ax.axhline(y=0.9, color='gray', linestyle='--', label='threshold=0.9')
    for bar, val in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f'{val:.4f}', ha='center', fontsize=10)
    ax.legend()
    plt.tight_layout()
    plt.savefig('figures/0225/uucat_results_6attacks.png', dpi=150)
    print('\nUCAT结果图已保存')