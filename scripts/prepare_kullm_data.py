import json
from pathlib import Path
from datasets import load_dataset
import random

def main():
    print("Loading kullm-v2 dataset...")
    ds = load_dataset('nlpai-lab/kullm-v2', split='train', streaming=True)
    
    out_dir = Path("data")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "kullm_v2_sample.jsonl"
    
    # We want 10,000 samples.
    # Kullm has instruction, input, and output.
    # We will format it as SLM expects: input (instruction) and output (response)
    
    count = 0
    max_records = 10000
    
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            # Construct a good prompt by combining instruction and input
            instruction = str(row.get("instruction", "")).strip()
            inp = str(row.get("input", "")).strip()
            response = str(row.get("output", "")).strip()
            
            if not instruction and not inp:
                continue
            if not response:
                continue
                
            combined_input = instruction
            if inp:
                combined_input += "\n" + inp
                
            out_row = {
                "input": combined_input,
                "output": response,
                "domain": "ko",
                "task_type": "korean"
            }
            
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            count += 1
            if count % 1000 == 0:
                print(f"[{count}/10000] Saved records...")
            if count >= max_records:
                break
                
    print(f"Finished saving {count} records to {out_path}.")

if __name__ == "__main__":
    main()
