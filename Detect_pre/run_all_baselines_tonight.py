import os
import subprocess
import time

# --- 1. 核心 VIP 名单 ---
vip_keywords = [
    "badnet", "wanet", "lira", "ftrojan", "sig", "blended", 
    "inputaware", "bpp", "refool", "trojannn", "ctrl", "ssba", "blind", "lf"
]

record_dir = "record"
all_folders = [d for d in os.listdir(record_dir) if os.path.isdir(os.path.join(record_dir, d))]

# 自动匹配最新文件夹
vip_attacks = []
for kw in vip_keywords:
    matches = [d for d in all_folders if kw in d.lower()]
    if matches:
        matches.sort()
        vip_attacks.append(matches[-1])

vip_attacks = list(set(vip_attacks))
print(f"⚡ 斩首行动启动！目标攻击数: {len(vip_attacks)}")

# --- 2. 调度策略 ---
# 第一优先级：AC 和 Spectral (推理快，数据含金量高)
# 第二优先级：NC (反向优化慢)
fast_methods = ["ac.py", "spectral.py"]

for attack in vip_attacks:
    for script in fast_methods:
        print(f"\n🔥 [TURBO RUN] {script} ON {attack}...")
        cmd = [
            "python", os.path.join("defense", script),
            "--result_file", attack,
            "--model", "resnet18",
            "--device", "cuda:0",
            "--num_workers", "0",
            "--batch_size", "1024" # 你设定的狂飙 Batch Size
        ]
        
        # 记录开始时间
        start = time.time()
        subprocess.run(cmd)
        print(f"✅ 完成 {script}, 耗时: {time.time()-start:.2f}s")

# --- 3. 如果还有时间，冲刺 NC ---
print("\n🦾 AC/Spectral 全部打完！开始冲刺重点 NC...")
priority_nc = ["wanet", "lira", "ftrojan", "badnet"] # 挑最值钱的 NC 跑
for attack in vip_attacks:
    if any(k in attack.lower() for k in priority_nc):
        print(f"\n🐢 [FINAL PUSH] nc.py ON {attack}...")
        subprocess.run([
            "python", "defense/nc.py",
            "--result_file", attack,
            "--model", "resnet18",
            "--device", "cuda:0",
            "--num_workers", "0",
            "--batch_size", "512" # NC 反向传播，512 较稳
        ])

print("\n🔋 任务已在停电前最大化压榨！数据已入库。")