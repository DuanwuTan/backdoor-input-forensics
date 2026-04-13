import os
import subprocess

# 1. 定义你要对比的防御算法（对应你截图的文件名）
defenses = ["ac", "spectral", "nc", "fp", "nad"]

# 2. 定义你已经训练好的攻击模型路径（根据你的 record 文件夹调整）
# 建议选出你最核心的几个攻击结果文件夹
attack_results = [
    "badnet_attack_1", 
    "wanet_attack_1", 
    "lira_attack_1", 
    "ftrojan_attack_final",
    "sig_attack_1",
    "blended_attack_1"
]

record_root = "./record" # 你的模型存放根目录
log_file = "baseline_overnight_log.txt"

with open(log_file, "w") as f:
    f.write("Starting Overnight Baseline Marathon...\n")

for attack in attack_results:
    attack_path = os.path.join(record_root, attack)
    if not os.path.exists(attack_path):
        print(f"Skipping {attack}, path not found.")
        continue

    for defense in defenses:
        print(f"--- Running {defense} on {attack} ---")
        
        # 构建命令 (BackdoorBench 标准调用格式)
        # 注意：有些防御可能需要额外参数，这里使用最通用的格式
        cmd = [
            "python", f"defense/{defense}.py", 
            "--result_path", attack_path
        ]
        
        try:
            # 执行并等待完成
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # 记录日志
            with open(log_file, "a") as f:
                f.write(f"\nRESULT: {defense} on {attack}\n")
                f.write(result.stdout[-500:]) # 只记录最后一部分输出（通常包含 AUC/ASR）
                if result.stderr:
                    f.write(f"\nERROR: {result.stderr}\n")
                    
        except Exception as e:
            print(f"Error running {defense} on {attack}: {e}")

print(f"All baselines finished! Check {log_file} tomorrow morning.")
