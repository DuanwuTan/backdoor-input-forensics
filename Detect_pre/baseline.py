import os

log_dir = "./baseline_logs"
results = {}

print("\n🔍 正在深度扫描日志文件内容...")

if not os.path.exists(log_dir):
    print(f"❌ 错误：找不到目录 {log_dir}")
else:
    for log_file in os.listdir(log_dir):
        if not log_file.endswith(".log"): continue
        
        path = os.path.join(log_dir, log_file)
        defense = log_file.split("_")[-1].replace(".log", "")
        
        # 识别攻击类型
        attack_type = "Unknown"
        for k in ['badnet', 'wanet', 'lira', 'ftrojan', 'sig', 'blended', 'bpp']:
            if k in log_file.lower():
                attack_type = k
                break
        
        if attack_type not in results: results[attack_type] = {}

        # 逐行读取，寻找数字
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line_lower = line.lower()
                
                # 针对 NC 的 Anomaly Index
                if "anomaly index" in line_lower:
                    try:
                        # 提取冒号后面的数字
                        value = line.split(":")[-1].strip()
                        # 只要前几个字符（防止后面带文字）
                        clean_value = "".join([c for c in value if c.isdigit() or c == '.'])
                        results[attack_type]['nc'] = clean_value
                        print(f"✅ 找到 NC 数据: {attack_type} -> {clean_value}")
                    except: pass
                
                # 针对 AC 和 Spectral 的 AUC
                if "auc" in line_lower:
                    try:
                        # 兼容 {'auc': 0.88} 或 auc: 0.88 格式
                        parts = line.replace("{","").replace("}","").replace("'","").replace("\"","").split(":")
                        for i, p in enumerate(parts):
                            if "auc" in p.lower() and i+1 < len(parts):
                                value = parts[i+1].split(",")[0].strip()
                                clean_value = "".join([c for c in value if c.isdigit() or c == '.'])
                                if clean_value:
                                    results[attack_type][defense] = clean_value
                                    print(f"✅ 找到 {defense} 数据: {attack_type} -> {clean_value}")
                    except: pass

# 打印最终汇总表
print("\n" + "="*60)
print("📊 --- UCAT 论文对比数据汇总表 ---")
print(f"{'Attack':<12} | {'AC (AUC)':<10} | {'Spec (AUC)':<10} | {'NC (Index)':<10}")
print("-" * 60)
for atk in sorted(results.keys()):
    vals = results[atk]
    ac = vals.get('ac', 'N/A')
    spec = vals.get('spectral', 'N/A')
    nc = vals.get('nc', 'N/A')
    print(f"{atk:<12} | {ac:<10} | {spec:<10} | {nc:<10}")
print("="*60)