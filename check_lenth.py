import json
import argparse
from transformers import AutoTokenizer

def check_length():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help="要扫描的数据集路径")
    parser.add_argument('--limit', type=int, default=18000, help="目标长度上限")
    parser.add_argument('--top_k_docs', type=int, default=80, help="保留的最多文档数 (和压测代码保持一致)")
    parser.add_argument('--num_samples', type=int, default=100, help="最多扫描几条")
    args = parser.parse_args()

    MODEL_PATH = "/home/yuan/saber/local_models/Qwen3-4B-Instruct-2507"
    
    print("🚀 正在加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    lengths = []
    valid_count = 0

    print(f"📊 开始极速抽样扫描 {args.dataset_path} ...")
    print(f"⚙️ 截断规则: Top-{args.top_k_docs} 篇文档 | 目标数量: 前 {args.num_samples} 条")
    
    # 🟢 智能识别文件格式：如果是 .json 就一口气全读，如果是 .jsonl 就一行行读
    test_pool = []
    with open(args.dataset_path, "r", encoding="utf-8") as f:
        if args.dataset_path.endswith('.json'):
            print(" └─ 检测到标准 .json 格式，正在加载整个文件...")
            test_pool = json.load(f)
        else:
            print(" └─ 检测到 .jsonl 格式，正在逐行解析...")
            for line in f:
                if line.strip():
                    try:
                        test_pool.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    # 开始抽样计算长度
    for i, data in enumerate(test_pool):
        # 🟢 急刹车：攒够100条立马停下！
        if len(lengths) >= args.num_samples:
            break
            
        question_str = data.get('question', data.get('query', ''))
        
        context_str = data.get('context', data.get('input', ''))
        if not context_str:
            ctxs = data.get('ctxs', data.get('docs', []))
            if not ctxs:
                ctxs.extend(data.get('positive_ctxs', []))
                ctxs.extend(data.get('hard_negative_ctxs', []))
                ctxs.extend(data.get('negative_ctxs', []))
            
            # 触发防 OOM 截断护盾
            ctxs = ctxs[:args.top_k_docs]
            
            context_pieces = []
            for doc_idx, c in enumerate(ctxs):
                title = c.get('title', '')
                text = c.get('text', '')
                context_pieces.append(f"Document [{doc_idx+1}]: {title}\n{text}")
            
            context_str = "\n\n".join(context_pieces)

        if not context_str: continue

        system_instruction = (
            "You are an expert reading comprehension assistant. "
            "Read the provided context carefully and answer the question based strictly on the context. "
            "Keep your answer as concise as possible. Do not include any conversational filler."
        )
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"# Context:\n{context_str}\n\n# Question:\n{question_str}"}
        ]
        
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        seq_len = len(tokenizer(prompt_text, add_special_tokens=False).input_ids)
        lengths.append(seq_len)
        
        if seq_len <= args.limit:
            valid_count += 1
            
        print(f" └─ 进度: [{len(lengths)}/{args.num_samples}] | 当前 Token 数: {seq_len}")

    if not lengths:
        print("⚠️ 未找到有效数据！请检查格式。")
        return

    print("\n" + "="*50)
    print(f"🎯 前 {args.num_samples} 条长度统计报告 (上限: {args.limit})")
    print("="*50)
    print(f"最大长度: {max(lengths)} Tokens")
    print(f"最小长度: {min(lengths)} Tokens")
    print(f"平均长度: {sum(lengths)//len(lengths)} Tokens")
    print(f"✅ 符合 <= {args.limit} 的样本数: {valid_count} (占比: {valid_count/len(lengths)*100:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    check_length()