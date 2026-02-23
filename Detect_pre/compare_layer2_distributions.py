import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.stats import permutation_test
import os

os.makedirs('figures/0222/comparison', exist_ok=True)

# 定义数据（从日志复制）
clean_pos = [(0,0)]*3 + [(0,19)]*1 + [(11,0)]*1 + [(12,11)]*1 + [(12,20)]*1 + [(19,12)]*1 + [(19,19)]*2 + [(19,20)]*2 + [(19,28)]*2 + [(20,0)]*1 + [(20,19)]*2 + [(20,20)]*2 + [(28,19)]*1
poison_pos = [(0,19)]*1 + [(11,0)]*1 + [(11,11)]*1 + [(12,0)]*2 + [(12,11)]*1 + [(19,11)]*1 + [(19,19)]*4 + [(19,20)]*2 + [(19,28)]*1 + [(20,12)]*1 + [(20,19)]*1 + [(20,20)]*1 + [(20,28)]*1 + [(28,11)]*1 + [(28,28)]*1

clean_pos = np.array(clean_pos)
poison_pos = np.array(poison_pos)

# 1. 绘制散点图
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(clean_pos[:,1], clean_pos[:,0], alpha=0.6, label='Clean', c='blue')
plt.xlim(0,31); plt.ylim(0,31)
plt.gca().invert_yaxis()  # 让原点在左上角
plt.title('Clean Samples')
plt.xlabel('x'); plt.ylabel('y')

plt.subplot(1,2,2)
plt.scatter(poison_pos[:,1], poison_pos[:,0], alpha=0.6, label='Poison', c='red')
plt.xlim(0,31); plt.ylim(0,31)
plt.gca().invert_yaxis()
plt.title('Poison Samples')
plt.xlabel('x'); plt.ylabel('y')
plt.tight_layout()
plt.savefig('figures/0222/comparison/scatter_comparison.png')
plt.show()

# 2. 绘制二维直方图对比（8x8网格）
def get_hist(pos, bins=8):
    H, xedges, yedges = np.histogram2d(pos[:,0], pos[:,1], bins=bins, range=[[0,32],[0,32]])
    return H / H.sum()  # 归一化

h_clean = get_hist(clean_pos, bins=8)
h_poison = get_hist(poison_pos, bins=8)

# 计算JS散度
m = 0.5 * (h_clean + h_poison)
js = 0.5 * (jensenshannon(h_clean.flatten(), m.flatten()) + jensenshannon(h_poison.flatten(), m.flatten()))
print(f"JS散度 (8x8网格): {js:.4f}")

# 3. 置换检验：比较质心距离图像中心(16,16)的欧氏距离
center = np.array([16,16])
clean_dist = np.linalg.norm(clean_pos - center, axis=1)
poison_dist = np.linalg.norm(poison_pos - center, axis=1)
obs_diff = np.mean(clean_dist) - np.mean(poison_dist)

def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

res = permutation_test((clean_dist, poison_dist), statistic, vectorized=True, n_resamples=9999, alternative='two-sided')
print(f"质心距离均值差: {obs_diff:.3f}, p-value: {res.pvalue:.4f}")

# 4. 保存坐标数据供后续使用
np.save('figures/0222/comparison/clean_pos.npy', clean_pos)
np.save('figures/0222/comparison/poison_pos.npy', poison_pos)