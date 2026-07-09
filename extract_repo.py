import os
import json
import glob
import re
import numpy as np
import pandas as pd

# ================= 配置区 =================
RESULTS_DIR = "repo_results"          # 存放你跑完的 jsonl 结果的文件夹
OUTPUT_CSV = "repo_summary.csv" # 最终汇总的 Excel/CSV 文件名
# ==========================================

def extract_repo_data():
    if not os.path.exists(RESULTS_DIR):
        print(f"❌ 找不到文件夹 '{RESULTS_DIR}'，请确保你已经跑完了实验。")
        return

    # 获取所有的 jsonl 文件
    jsonl_files = glob.glob(os.path.join(RESULTS_DIR, "*.jsonl"))
    
    if not jsonl_files:
        print(f"❌ 在 '{RESULTS_DIR}' 目录下为空。")
        return

    print(f"📂 找到 {len(jsonl_files)} 个结果文件，正在提取 RepoBench 数据...")
    all_data = []

    for filepath in jsonl_files:
        # 提取 retain_ratio，兼容 r0.1 和 r0.10 的命名
        match = re.search(r'_r([0-9.]+)_', filepath)
        if not match:
            continue
        retain_ratio = float(match.group(1))
        
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
                                "em_score": [], "es_score": [], 
                                "payload_mb": [], "ttft_ms": [], "tpot_ms": []
                            }
                            
                        # 收集所有的分数和系统开销
                        file_metrics[method]["em_score"].append(metrics.get("em_score", 0.0))
                        file_metrics[method]["es_score"].append(metrics.get("es_score", 0.0))
                        file_metrics[method]["payload_mb"].append(metrics.get("payload_mb", 0.0))
                        file_metrics[method]["ttft_ms"].append(metrics.get("ttft_ms", 0.0))
                        file_metrics[method]["tpot_ms"].append(metrics.get("tpot_ms", 0.0))
                except json.JSONDecodeError:
                    continue
        
        # 计算该 ratio 下各个方法的均值
        for method, metrics_list in file_metrics.items():
            if not metrics_list["em_score"]: continue
            
            all_data.append({
                "Retain Ratio": retain_ratio,
                "Method": method,
                "EM (%)": np.mean(metrics_list["em_score"]) * 100,  # 转百分比
                "ES (%)": np.mean(metrics_list["es_score"]) * 100,  # 转百分比
                "Payload (MB)": np.mean(metrics_list["payload_mb"]),
                "Avg TTFT (ms)": np.mean(metrics_list["ttft_ms"]),
                "Avg TPOT (ms)": np.mean(metrics_list["tpot_ms"]),
                "Samples": len(metrics_list["em_score"])
            })

    if not all_data:
        print("❌ 提取失败，文件内可能没有有效数据。")
        return

    # 转换为 DataFrame 并进行美化排序
    df = pd.DataFrame(all_data)
    df = df.sort_values(by=["Method", "Retain Ratio"], ascending=[True, False]).reset_index(drop=True)
    
    # 统一保留两位小数
    df = df.round({
        "EM (%)": 2, "ES (%)": 2,
        "Payload (MB)": 2, "Avg TTFT (ms)": 1, "Avg TPOT (ms)": 1
    })

    # 保存并打印
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("\n" + "="*95)
    print(f"📊 RepoBench 实验汇总结果 (已保存至 {OUTPUT_CSV})")
    print("="*95)
    print(df.to_string(index=False))
    print("="*95)

if __name__ == "__main__":
    extract_repo_data()