import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--lora_path", type=str, default="artifacts_qlora")
    parser.add_argument("--prompt", type=str, default="분산 시스템에서 CAP 정리를 2문장으로 설명해줘")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Base Model in 4-bit (VRAM optimized)
    print(f"Loading Base 4-bit Model: {args.base_model}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 2. Attach PEFT (LoRA) weights if available
    try:
        print(f"Loading LoRA weights from: {args.lora_path}...")
        model = PeftModel.from_pretrained(model, args.lora_path)
    except Exception as e:
        print(f"Could not load LoRA (Proceeding with Base Model): {e}")

    # 3. Format Prompt & Generate (Must match training format)
    inp = args.prompt
    text = f"### Instruction\n{inp}\n\n### Response\n"
    print("\n--- Generating ---")
    
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256,
            temperature=0.3, # Low temperature for factual knowledge
            do_sample=True,
            top_p=0.9
        )
        
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n[AI 답변]:")
    print(response.strip())

if __name__ == "__main__":
    main()
