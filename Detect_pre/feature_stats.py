import numpy as np
from scipy import stats
import os
import pandas as pd

def statistical_analysis():
    """
    计算特征的均值、标准差，进行独立t检验
    """
    
    # =====================
    # 加载特征文件（修正路径）
    # =====================
    poison_path = "features_badnet_poison.npy"
    clean_path = "features_badnet_clean.npy"
    
    if not os.path.exists(poison_path):
        print(f"错误: 文件不存在 {poison_path}")
        return
    if not os.path.exists(clean_path):
        print(f"错误: 文件不存在 {clean_path}")
        return
    
    print("加载特征文件...")
    features_poison = np.load(poison_path)
    features_clean = np.load(clean_path)
    
    print(f"后门特征形状: {features_poison.shape}")
    print(f"干净特征形状: {features_clean.shape}")
    
    # =====================
    # 特征名称
    # =====================
    feature_names = ['面积', '凸包比', '离心率', '质心x', '质心y', '熵']
    
    # =====================
    # 计算统计量
    # =====================
    print("\n计算统计量...")
    
    results = []
    
    for i, name in enumerate(feature_names):
        poison_data = features_poison[:, i]
        clean_data = features_clean[:, i]
        
        poison_mean = poison_data.mean()
        poison_std = poison_data.std()
        clean_mean = clean_data.mean()
        clean_std = clean_data.std()
        
        t_stat, p_value = stats.ttest_ind(poison_data, clean_data)
        
        # Cohen's d 效应大小
        pooled_std = np.sqrt(((len(poison_data)-1)*poison_std**2 + 
                              (len(clean_data)-1)*clean_std**2) / 
                             (len(poison_data) + len(clean_data) - 2))
        cohens_d = (poison_mean - clean_mean) / (pooled_std + 1e-10)
        
        results.append({
            '特征': name,
            '后门均值': poison_mean,
            '后门标准差': poison_std,
            '干净均值': clean_mean,
            '干净标准差': clean_std,
            't统计量': t_stat,
            'p值': p_value,
            "Cohen's d": cohens_d,
            '显著性': '***' if p_value < 0.001 else ('**' if p_value < 0.01 else ('*' if p_value < 0.05 else 'ns'))
        })
        
        print(f"\n{name}:")
        print(f"  后门: μ={poison_mean:.6f}, σ={poison_std:.6f}")
        print(f"  干净: μ={clean_mean:.6f}, σ={clean_std:.6f}")
        print(f"  t统计量: {t_stat:.4f}, p值: {p_value:.2e}")
        print(f"  Cohen's d: {cohens_d:.4f}")
    
    # =====================
    # 创建DataFrame
    # =====================
    df = pd.DataFrame(results)
    
    # =====================
    # 生成Markdown表格
    # =====================
    markdown_content = generate_markdown_table(df)
    
    # =====================
    # 保存到文件
    # =====================
    output_dir = "./exp_logs"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "0214_stats.md")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"\n✓ 统计结果已保存: {output_path}")
    
    # =====================
    # 打印Markdown表格
    # =====================
    print("\n" + "="*100)
    print("Markdown格式表格:")
    print("="*100)
    print(markdown_content)

def generate_markdown_table(df):
    """
    生成Markdown格式的表格
    """
    header = """# BadNet后门检测特征统计分析报告

## 数据集信息
- 后门样本数: 500
- 干净样本数: 500
- 特征维度: 6

## 统计结果

### 注释说明
- ***：p < 0.001（极显著）
- **：p < 0.01（非常显著）
- *：p < 0.05（显著）
- ns：p ≥ 0.05（不显著）
- Cohen's d：效应大小（|d| > 0.8为大效应）

### 详细统计表

"""
    
    # 创建表格
    table_header = "| 特征 | 后门均值 | 后门标准差 | 干净均值 | 干净标准差 | t统计量 | p值 | Cohen's d | 显著性 |\n"
    table_sep = "|------|---------|----------|---------|----------|--------|-----|-----------|--------|\n"
    
    # 生成表格行
    table_rows = ""
    for _, row in df.iterrows():
        table_rows += "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2e} | {:.4f} | {} |\n".format(
            row['特征'], 
            row['后门均值'], 
            row['后门标准差'], 
            row['干净均值'], 
            row['干净标准差'], 
            row['t统计量'], 
            row['p值'], 
            row["Cohen's d"], 
            row['显著性']
        )
    
    # 统计显著特征
    significant = df[df['显著性'] != 'ns']
    
    summary = f"""
## 摘要

### 显著特征统计
- 总特征数: {len(df)}
- 显著特征数: {len(significant)} ({len(significant)/len(df)*100:.1f}%)
- 极显著特征数(p<0.001): {len(df[df['显著性']=='***'])}

### 大效应特征(|Cohen's d| > 0.8)
"""
    
    large_effect = df[abs(df["Cohen's d"]) > 0.8]
    if len(large_effect) > 0:
        for _, row in large_effect.iterrows():
            summary += "- {}: d = {:.4f}\n".format(row['特征'], row["Cohen's d"])
    else:
        summary += "- 无\n"
    
    markdown = header + table_header + table_sep + table_rows + summary
    return markdown

if __name__ == "__main__":
    statistical_analysis()