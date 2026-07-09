"""
Parquet to JSONL Converter for RULER Benchmark
"""
import pandas as pd

def convert_parquet(parquet_path, output_jsonl_path):
    print(f"📦 Loading Parquet file: {parquet_path}")
    
    # Load the parquet file into a pandas DataFrame
    df = pd.read_parquet(parquet_path)
    
    print("\n📊 DATA SCHEMA:")
    for col, dtype in df.dtypes.items():
        print(f"  └─ {col}: {dtype}")
        
    print(f"\nTotal Rows: {len(df):,}")
    
    # Preview the first row to understand the structure
    print("\n👀 PREVIEW (Row 1):")
    first_row = df.iloc[0].to_dict()
    for key, val in first_row.items():
        # Truncate long text for the preview
        val_str = str(val)
        display_val = (val_str[:80] + '...') if len(val_str) > 80 else val_str
        print(f"  └─ {key}: {display_val}")
    
    # Export to JSONL format for your benchmark pipeline
    print(f"\n⚙️ Converting to JSONLines...")
    df.to_json(output_jsonl_path, orient="records", lines=True)
    
    print(f"✅ Conversion complete! Saved to: {output_jsonl_path}")

if __name__ == "__main__":
    PARQUET_FILE = "../data/benchmarks/RULER/test-00000-of-00002.parquet"
    OUTPUT_FILE = "../data/benchmarks/RULER/ruler_test_16k.jsonl"
    
    convert_parquet(PARQUET_FILE, OUTPUT_FILE)