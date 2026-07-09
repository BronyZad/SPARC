import json
import os

# 你的 6 个待排查的 Judge 文件
FILES = [
    "/home/yuan/saber/code/graded_saber_checkpoint_r0.1_ruler_qa_1.jsonl",
    "/home/yuan/saber/code/graded_saber_checkpoint_r0.1_ruler_qa_2.jsonl",
    "/home/yuan/saber/code/graded_saber_longbench_r0.1_2wikimqa_e.jsonl",
    "/home/yuan/saber/code/graded_saber_longbench_r0.1_hotpotqa_e.jsonl",
    "/home/yuan/saber/code/graded_saber_longbench_r0.1_multifieldqa_en_e.jsonl",
    "/home/yuan/saber/code/graded_saber_longbench_r0.1_qasper_e.jsonl"
]

# 我们拿你的主力方法 Saber-BIC 来做抽查
METHOD = "Saber-BIC"

print("="*70)
print(" 🕵️‍♂️ 裁判放水调查局 v2.0：专抓『废话连篇却拿高分』的宽松误判")
print("="*70)

for fp in FILES:
    if not os.path.exists(fp):
        continue
    
    task_name = os.path.basename(fp).replace("graded_", "").replace(".jsonl", "")
    print(f"\n📁 正在排查任务: {task_name}")
    
    suspect_cases_found = 0
    with open(fp, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                rec = json.loads(line)
                
                # 获取 LLM 裁判打分
                score_info = rec.get("llm_judge_scores", {}).get(METHOD, {})
                raw_score = score_info.get("score", 0)
                judgment = score_info.get("judgment", "无裁判评语 (可能是正则秒判)")
                
                # 智能归一化分数 (兼容 1分制, 10分制, 100分制)，统一转为百分制来判断
                if 0 < raw_score <= 1.0:
                    norm_score = raw_score * 100
                elif 1.0 < raw_score <= 10.0:
                    norm_score = raw_score * 10
                else:
                    norm_score = raw_score
                
                # 获取预测和答案
                res = rec.get("results", {}).get(METHOD, {})
                prediction = str(res.get("cleaned_text", res.get("generated_text", "")))
                references = rec.get("outputs", rec.get("reference", rec.get("ground_truth", rec.get("answers", []))))
                
                if isinstance(references, list) and references:
                    ref_str = str(references[0])
                else:
                    ref_str = str(references)
                
                # 🚨 核心抓虫逻辑：
                # 1. 裁判给了高分 (>= 80分，也就是 10分制的 8分以上)
                # 2. 预测长度比标准答案多出至少 80 个字符 (意味着掺杂了极大量废话/幻觉)
                is_loose_score = norm_score >= 80
                is_fluff = len(prediction) > (len(ref_str) + 80)
                
                if is_loose_score and is_fluff:
                    print("-" * 50)
                    print(f"🚨 抓到『高分放水』嫌疑犯！ (裁判给了: {raw_score} 分)")
                    print(f"✅ 真实答案 (长度 {len(ref_str)}): {ref_str}")
                    print(f"🤖 冗长预测 (长度 {len(prediction)}): {prediction[:200]} ... (截断)")
                    print(f"⚖️ 裁判评语: {judgment.strip()}")
                    suspect_cases_found += 1
                    
                    # 每个文件抽查前 3 个放水案例
                    if suspect_cases_found >= 3:
                        break
                        
            except Exception as e:
                pass
                
    if suspect_cases_found == 0:
        print(" -> ✅ 这个任务没有发现明显的『写小作文还能拿高分』现象。")

print("\n" + "="*70)
print(" 🏁 排查完毕！请检查上述输出，看看裁判是怎么『和稀泥』的。")