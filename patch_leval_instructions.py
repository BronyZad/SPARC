import json
import os
import glob

# 🟢 Directories
CHECKPOINT_DIR = "./"
LEVAL_DIR = "../data/benchmarks/LEval"

def patch_checkpoints():
    print("🔍 Mapping original LEval datasets...")
    original_files = {}
    for root, _, files in os.walk(LEVAL_DIR):
        for f in files:
            if f.endswith(".jsonl"):
                original_files[f] = os.path.join(root, f)

    checkpoints = glob.glob(os.path.join(CHECKPOINT_DIR, "saber_checkpoint_*.jsonl"))
    
    if not checkpoints:
        print("⚠️ No checkpoints found in the current directory.")
        return

    for ckpt in checkpoints:
        # Match the checkpoint to the original source file
        matched_orig = None
        for orig_name in original_files.keys():
            if ckpt.endswith(orig_name):
                matched_orig = original_files[orig_name]
                break
        
        if not matched_orig:
            print(f"⚠️ Could not find original dataset for {ckpt}. Skipping.")
            continue
            
        # Load the original LEval data
        with open(matched_orig, "r", encoding="utf-8") as f:
            orig_data = [json.loads(line) for line in f]
            
        patched_records = []
        updates_made = 0
        
        with open(ckpt, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): 
                    continue
                record = json.loads(line)
                
                # If it already has an instruction (e.g., from a recent run), skip patching
                if "instruction" not in record:
                    doc_id = record.get("id", "")
                    try:
                        # Extract index from ID (e.g., "doc_4_abc123" -> 4)
                        idx = int(doc_id.split("_")[1])
                        instructions = orig_data[idx].get("instructions", [])
                        
                        # Inject the instruction
                        record["instruction"] = instructions[0] if instructions else "Answer based on the document."
                        updates_made += 1
                    except Exception as e:
                        print(f"  ❌ Error parsing ID {doc_id} in {ckpt}: {e}")
                
                patched_records.append(record)
                
        # Safely overwrite the checkpoint with the patched data
        if updates_made > 0:
            with open(ckpt, "w", encoding="utf-8") as f:
                for rec in patched_records:
                    f.write(json.dumps(rec) + "\n")
            print(f"✅ Patched {updates_made} records in {os.path.basename(ckpt)}")
        else:
            print(f"⏭️ No updates needed for {os.path.basename(ckpt)}")

if __name__ == "__main__":
    patch_checkpoints()