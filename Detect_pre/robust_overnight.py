import os
import time

# --- 配置区 ---
defenses = ["ac", "spectral", "nc"]
model_type = "resnet18"
record_root = "./record"
log_dir = "./baseline_logs"
os.makedirs(log_dir, exist_ok=True)

# --- 自动扫描所有包含 attack_result.pt 的文件夹 ---
valid_attacks = []
for folder in os.listdir(record_root):
    folder_path = os.path.join(record_root, folder)
    if os.path.isdir(folder_path):
        # 检查里面有没有 attack_result.pt
        if os.path.exists(os.path.join(folder_path, "attack_result.pt")):
            valid_attacks.append(folder)

print(f"🔍 Found {len(valid_attacks)} valid attack folders: {valid_attacks}")

# --- 执行区 ---
for attack in valid_attacks:
    for defense in defenses:
        log_file = f"{log_dir}/{attack}_{defense}.log"
        print(f"🚀 Running {defense} on {attack}...")
        
        # 这里的 --result_file 只传文件夹名，BackdoorBench 会自己拼 record/ 和 /attack_result.pt
        cmd = (
            f"python defense/{defense}.py "
            f"--result_file {attack} "
            f"--model {model_type} "
            f"--batch_size 512 "
            f"> {log_file} 2>&1"
        )
        
        start_time = time.time()
        os.system(cmd)
        duration = (time.time() - start_time) / 60
        print(f"✅ Done in {duration:.2f} min.")

print("🏁 ALL TASKS COMPLETED. See you tomorrow!")