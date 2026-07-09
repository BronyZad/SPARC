import os
import json
import glob
import re
import numpy as np
import pandas as pd

# ================= 配置区 =================
RESULTS_DIR = "lcc_results"          # 你存放 jsonl 文件的文件夹
OUTPUT_CSV = "lcc_summary.csv" # 最终生成的汇总表格文件名
# ==========================================

def extract_lcc_data():
    if not os.path.exists(RESULTS_DIR):
        print(f"❌ 找不到文件夹 '{RESULTS_DIR}'，请检查路径。")
        return

    # 寻找所有的 jsonl 文件
    file_patterns = os.path.join(RESULTS_DIR, "*.jsonl")
    jsonl_files = glob.glob(file_patterns)
    
    if not jsonl_files:
        print(f"❌ 在 '{RESULTS_DIR}' 目录下没有找到任何 .jsonl 文件。")
        return

    print(f"📂 找到 {len(jsonl_files)} 个结果文件，正在提取数据...")
    
    all_data = []

    for filepath in jsonl_files:
        # 使用正则从文件名中提取 retain_ratio (例如从 saber_checkpoint_r0.05_lcc_e.jsonl 提取 0.05)
        match = re.search(r'_r([0-9.]+)_', filepath)
        if not match:
            print(f"⚠️ 无法从文件名 {os.path.basename(filepath)} 提取 retain_ratio，已跳过。")
            continue
        
        retain_ratio = float(match.group(1))
        
        # 用于临时存储当前文件中各方法的数据
        file_metrics = {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    record = json.loads(line)
                    results = record.get("results", {})
                    
                    for method, metrics in results.items():
                        if not metrics.get("success"): continue
                        
                        if method not in file_metrics:
                            file_metrics[method] = {
                                "em_score": [], "payload_mb": [], 
                                "ttft_ms": [], "tpot_ms": []
                            }
                            
                        file_metrics[method]["em_score"].append(metrics.get("em_score", 0.0))
                        file_metrics[method]["payload_mb"].append(metrics.get("payload_mb", 0.0))
                        file_metrics[method]["ttft_ms"].append(metrics.get("ttft_ms", 0.0))
                        file_metrics[method]["tpot_ms"].append(metrics.get("tpot_ms", 0.0))
                        
                except json.JSONDecodeError:
                    continue
        
        # 计算当前 ratio 下，各个方法的平均值
        for method, metrics_list in file_metrics.items():
            if not metrics_list["em_score"]: continue
            
            all_data.append({
                "Retain Ratio": retain_ratio,
                "Method": method,
                "Edit Sim (%)": np.mean(metrics_list["em_score"]) * 100, # 转换为百分比
                "Payload (MB)": np.mean(metrics_list["payload_mb"]),
                "Avg TTFT (ms)": np.mean(metrics_list["ttft_ms"]),
                "Avg TPOT (ms)": np.mean(metrics_list["tpot_ms"]),
                "Samples": len(metrics_list["em_score"])
            })

    if not all_data:
        print("❌ 没有提取到任何有效数据。")
        return

    # 转换为 DataFrame 并进行排序和格式化
    df = pd.DataFrame(all_data)
    
    # 按照 Method 和 Retain Ratio 进行排序，方便观察趋势
    df = df.sort_values(by=["Method", "Retain Ratio"], ascending=[True, False]).reset_index(drop=True)
    
    # 保留一位或两位小数，让表格更整洁
    df = df.round({
        "Edit Sim (%)": 2, 
        "Payload (MB)": 2, 
        "Avg TTFT (ms)": 1, 
        "Avg TPOT (ms)": 1
    })

    # 保存到 CSV
    df.to_csv(OUTPUT_CSV, index=False)
    
    # 在终端打印出漂亮的表格
    print("\n" + "="*85)
    print(f"📊 LCC 实验汇总结果 (已保存至 {OUTPUT_CSV})")
    print("="*85)
    print(df.to_string(index=False))
    print("="*85)

if __name__ == "__main__":
    extract_lcc_data()