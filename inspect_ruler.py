import json
import os
import glob

ruler_dir = "../data/benchmarks/RULER/"
jsonl_files = glob.glob(os.path.join(ruler_dir, "ruler_test*.jsonl"))
print(jsonl_files)
if not jsonl_files:
    print(f"❌ No .jsonl files found in {ruler_dir}")

for file_path in jsonl_files:
    task_name = os.path.basename(file_path)
    # Skip the massive test splits just to look at the task structures
    if "test" in task_name: 
        continue
        
    print(f"\n{'='*80}")
    print(f"🔎 Task: {task_name}")
    print(f"{'='*80}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line:
                data = json.loads(first_line)
                input_text = data.get('input', '')
                outputs = data.get('outputs', [])
                
                print(f"🛑 EXPECTED OUTPUTS: {outputs}\n")
                
                # The actual instruction usually lives at the very end of the massive context
                tail_length = min(800, len(input_text))
                print(f"🔚 PROMPT TAIL (last {tail_length} chars):")
                print(f"...{input_text[-tail_length:]}")
                
    except Exception as e:
        print(f"⚠️ Error reading {task_name}: {e}")