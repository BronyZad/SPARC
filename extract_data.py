import os
import json
import pandas as pd
import numpy as np
import re
import string

# ==========================================
# ⚙️ 1. 全局配置与大一统矩阵
# ==========================================
CODE_DIR = "/home/yuan/saber/code"
METHODS = ["Native-Baseline", "Uniform-INT4", "ablation_inverted", "Saber-BIC", "SnapKV"]
COLUMNS = ["Task", "Metric"] + METHODS

# 统一整合的超级任务字典 {Benchmark: {Task: Metric_Type}}
TASKS_MATRIX = {
    "LongBench": {
        "multi_news_e": "ROUGE", "gov_report_e": "ROUGE", "samsum_e": "ROUGE",
        # 🚀 🛠️ 修正点 1：将 4 个客观/计数任务由 "JUDGE" 修正为 "EM"，彻底根治 N/A 与 1% 乌龙
        "passage_count_e": "EM", "passage_retrieval_en_e": "EM", 
        "trec_e": "EM", "triviaqa_e": "EM",
        # 仅有这 4 个主观长文本问答保留大模型裁判模式
        "2wikimqa_e": "JUDGE", "hotpotqa_e": "JUDGE", "multifieldqa_en_e": "JUDGE", "qasper_e": "JUDGE"
    },
    "LEval": {
        t: "JUDGE" for t in ["coursera", "financial_qa", "gov_report_summ", "gsm100", 
                             "legal_contract_qa", "meeting_summ", "multidoc_qa", "narrative_qa", 
                             "natural_question", "news_summ", "paper_assistant", "quality", 
                             "review_summ", "sci_fi", "scientific_qa", "topic_retrieval_longchat", 
                             "tpo", "tv_show_summ"]
    },
    "RULER": {
        "cwe": "EM", "fwe": "EM", "niah_multiquery": "EM", "niah_multivalue": "EM", "vt": "EM",
        "qa_1": "JUDGE", "qa_2": "JUDGE", "test_8k": "EM", "test_16k": "EM"
    }
}

PERF_METRICS_LIST = ["Payload(MB)", "Prefill(ms)", "Avg TTFT(ms)", "P95 TTFT(ms)", "Avg TPOT(ms)", "P95 TPOT(ms)"]

# ==========================================
# 🔍 2. 底层雷达系统 (文件寻址)
# ==========================================
def find_target_file(task_name, require_graded=False):
    task_lower = task_name.lower()
    possible_files = []
    for root, dirs, files in os.walk(CODE_DIR):
        for file in files:
            f_lower = file.lower()
            if not f_lower.endswith(".jsonl"): continue
            
            # 精准匹配任务名
            if re.search(rf"{task_lower}(?![a-z0-9])", f_lower):
                if require_graded and "graded" in f_lower:
                    possible_files.append(os.path.join(root, file))
                elif not require_graded and "graded" not in f_lower:
                    possible_files.append(os.path.join(root, file))
                    
    if possible_files:
        possible_files.sort(key=os.path.getmtime, reverse=True) 
        return possible_files[0]
    return None

# ==========================================
# 🧮 3. 辅助计算 (为客观题提供精准本地算分)
# ==========================================
def normalize_text(text):
    if not isinstance(text, str): text = str(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return " ".join(text.split())

def check_exact_match(prediction, references):
    if not prediction: return 0.0
    pred_norm = normalize_text(prediction)
    for ref in references:
        ref_norm = normalize_text(ref)
        # 🚀 🛠️ 修正点 2：引入 \b 单词边界匹配，防止数字（如5和15）相互串岗误判
        if ref_norm and re.search(rf'\b{re.escape(ref_norm)}\b', pred_norm):
            return 100.0
    return 0.0

# ==========================================
# 🧠 4. 核心解析引擎 (准确率 + 性能)
# ==========================================
def parse_accuracy(filepath, metric_type):
    if not filepath or not os.path.exists(filepath): 
        return {m: "N/A" for m in METHODS}
    temp = {m: [] for m in METHODS}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                rec = json.loads(line)
                
                for m in METHODS:
                    if metric_type == "JUDGE":
                        s = rec.get("llm_judge_scores", {}).get(m, {}).get("score")
                        if s is not None: temp[m].append(s * 10) 
                    else:
                        res = rec.get("results", {}).get(m, {})
                        if res.get("success", True):
                            if metric_type == "ROUGE":
                                r = res.get("rouge_score", res.get("rouge", res.get("rouge-l")))
                                if r is not None: temp[m].append(r * 100)
                            
                            elif metric_type == "EM":
                                prediction = res.get("cleaned_text", res.get("generated_text", ""))
                                references = rec.get("outputs", rec.get("reference", rec.get("ground_truth", rec.get("answers", []))))
                                if isinstance(references, str): references = [references]
                                
                                score = check_exact_match(prediction, references)
                                temp[m].append(score)
            except Exception: pass
            
    return {m: round(np.mean(temp[m]), 2) if temp[m] else "N/A" for m in METHODS}

def parse_performance(filepath):
    data_vault = {m: {"payload": [], "prefill": [], "ttft": [], "tpot": []} for m in METHODS}
    
    if filepath and os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    rec = json.loads(line)
                    for m in METHODS:
                        res = rec.get("results", {}).get(m, {})
                        if res.get("success", True):
                            payload = res.get("payload_mb", res.get("payload"))
                            prefill = res.get("prefill_time", res.get("prefill_ms", res.get("prefill")))
                            ttft = res.get("ttft", res.get("ttft_ms"))
                            tpot = res.get("tpot", res.get("tpot_ms"))
                            
                            if payload is not None: data_vault[m]["payload"].append(payload)
                            if prefill is not None: data_vault[m]["prefill"].append(prefill)
                            if ttft is not None: data_vault[m]["ttft"].append(ttft)
                            if tpot is not None: data_vault[m]["tpot"].append(tpot)
                except Exception: pass
    
    results = {p: {} for p in PERF_METRICS_LIST}
    for m in METHODS:
        v = data_vault[m]
        results["Payload(MB)"][m] = round(np.mean(v["payload"]), 2) if v["payload"] else "N/A"
        results["Prefill(ms)"][m] = round(np.mean(v["prefill"]), 2) if v["prefill"] else "N/A"
        results["Avg TTFT(ms)"][m] = round(np.mean(v["ttft"]), 2) if v["ttft"] else "N/A"
        results["P95 TTFT(ms)"][m] = round(np.percentile(v["ttft"], 95), 2) if v["ttft"] else "N/A"
        results["Avg TPOT(ms)"][m] = round(np.mean(v["tpot"]), 2) if v["tpot"] else "N/A"
        results["P95 TPOT(ms)"][m] = round(np.percentile(v["tpot"], 95), 2) if v["tpot"] else "N/A"
    return results

# ==========================================
# 🏭 5. 大一统流水线装配
# ==========================================
def build_combined_tables():
    for benchmark_name, task_dict in TASKS_MATRIX.items():
        print(f"\n🔄 正在生成 {benchmark_name} 全面整合表...")
        records = []
        
        for task, m_type in task_dict.items():
            print(f"  -> 处理任务: {task:<20} [{m_type}]")
            
            acc_file = find_target_file(task, require_graded=(m_type == "JUDGE"))
            acc_scores = parse_accuracy(acc_file, m_type)
            
            acc_metric_name = "LLM-Judge" if m_type == "JUDGE" else ("ROUGE" if m_type == "ROUGE" else "ExactMatch")
            records.append({"Task": task, "Metric": acc_metric_name, **acc_scores})
            
            perf_file = find_target_file(task, require_graded=False)
            perf_scores = parse_performance(perf_file)
            
            for p_metric in PERF_METRICS_LIST:
                is_all_na = all(val == "N/A" for val in perf_scores[p_metric].values())
                if not is_all_na:
                    records.append({"Task": task, "Metric": p_metric, **perf_scores[p_metric]})
                
        out_path = os.path.join(CODE_DIR, f"Combined_{benchmark_name}.csv")
        pd.DataFrame(records, columns=COLUMNS).to_csv(out_path, index=False)
        print(f"✅ {benchmark_name} 生成完毕 -> {out_path}")

if __name__ == "__main__":
    print("=" * 70)
    print(" 🚀 大模型基准评测：【终极单表合并 + 智能容错】提取引擎")
    print("=" * 70)
    build_combined_tables()
    print("=" * 70)
    print(" 🎉 搞定！已生成 3 个 Combined_XXX.csv")