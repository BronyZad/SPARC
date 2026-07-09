import json
import glob
import pandas as pd
import os
import numpy as np

# 1. 锁定包含所有 checkpoint 的文件夹
result_dir = "/home/yuan/saber/code/repo_results/"
checkpoint_files = glob.glob(os.path.join(result_dir, "saber_checkpoint_r*_repobench.jsonl"))

all_metrics = []

print(f"🔍 找到了 {len(checkpoint_files)} 个 RepoBench 数据文件，开始提取...")

# 2. 遍历提取数据
for file_path in checkpoint_files:
    # 从文件名中提取 retention_ratio，例如提取 '0.01'
    filename = os.path.basename(file_path)
    ratio_str = filename.split('_r')[1].split('_')[0]
    
    methods_data = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            
            for method, metrics in record.get("results", {}).items():
                if method not in methods_data:
                    methods_data[method] = {"em": [], "payload": [], "ttft": []}
                
                if metrics.get("success"):
                    # 假设 RepoBench 也有类似的 EM 和 TTFT 字段，如果有出入下面可能需要微调
                    methods_data[method]["em"].append(metrics.get("em_score", 0))
                    methods_data[method]["payload"].append(metrics.get("payload_mb", 0))
                    methods_data[method]["ttft"].append(metrics.get("ttft_ms", 0))
    
    # 3. 计算每个流派的平均值
    for method, data in methods_data.items():
        if len(data["em"]) == 0: continue
        all_metrics.append({
            "Ratio": ratio_str,
            "Method": method,
            "Accuracy": np.mean(data["em"]),
            "Payload(MB)": np.mean(data["payload"]),
            "TTFT(ms)": np.mean(data["ttft"])
        })

# 4. 保存为 CSV
if all_metrics:
    df = pd.DataFrame(all_metrics)
    # 按比例和方法排序，让表格更好看
    df = df.sort_values(by=["Ratio", "Method"])
    out_path = "/home/yuan/saber/code/Combined_RepoBench.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ 提取成功！已生成汇总表格: {out_path}")
else:
    print("⚠️ 提取失败，未能从文件中解析出有效数据，请检查 JSON 字段名。")