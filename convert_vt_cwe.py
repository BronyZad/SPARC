"""
RULER Benchmark Converter (VT and CWE)
Converts specific Parquet files to JSONLines format.
"""
import pandas as pd
import os

def convert_parquet_to_jsonl(parquet_path, jsonl_path):
    if not os.path.exists(parquet_path):
        print(f"❌ Error: File not found at {parquet_path}")
        return

    print(f"📦 Loading Parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    print(f"📊 Found {len(df):,} rows.")
    
    # Check if 'task' column exists, print unique tasks just to be sure
    if 'task' in df.columns:
        tasks = df['task'].unique()
        print(f"🔍 Tasks included: {', '.join(tasks)}")
        
    print(f"⚙️ Converting to JSONLines...")
    df.to_json(jsonl_path, orient="records", lines=True)
    print(f"✅ Saved to: {jsonl_path}\n" + "-"*40)

if __name__ == "__main__":
    # Base directory relative to ~/saber/code/
    base_dir = "../data/benchmarks/RULER/"
    
    files_to_convert = [
        {
            "in": os.path.join(base_dir, "niah_multiquery-00000-of-00001.parquet"),
            "out": os.path.join(base_dir, "ruler_niah_multiquery.jsonl")
        },
        {
            "in": os.path.join(base_dir, "niah_multivalue-00000-of-00001.parquet"),
            "out": os.path.join(base_dir, "ruler_niah_multivalue.jsonl")
        }
    ]

    for files in files_to_convert:
        convert_parquet_to_jsonl(files["in"], files["out"])
