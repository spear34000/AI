import os
import torch
import argparse
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--data_path", type=str, default="data/kullm_v2_subset.jsonl")
    parser.add_argument("--output_dir", type=str, default="artifacts_qlora")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    return parser.parse_args()

def formatting_prompts_func(example):
    output_texts = []
    # Use batched if available, or just single mapping
    if isinstance(example.get('instruction', ''), list):
        for i in range(len(example.get('instruction', []))):
            inp = example.get('instruction', example.get('input', []))[i]
            out = example.get('output', example.get('response', []))[i]
            text = f"### Instruction\n{inp}\n\n### Response\n{out}"
            output_texts.append(text)
    else:
        inp = example.get('instruction', example.get('input', ''))
        out = example.get('output', example.get('response', ''))
        text = f"### Instruction\n{inp}\n\n### Response\n{out}"
        return text

def main():
    args = parse_args()
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print(f"Loading Base Model ({args.model_id}) in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = prepare_model_for_kbit_training(model)
    
    print("Injecting LoRA adapters...")
    peft_config = LoraConfig(
        r=8, 
        lora_alpha=16, 
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    print(f"Loading Dataset {args.data_path}...")
    dataset = load_dataset("json", data_files=args.data_path, split="train")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        max_steps=500, # Increased for real training
        save_steps=100,
        optim="paged_adamw_8bit",
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none"
    )
    
    print("Starting QLoRA Fine-tuning!")
    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            formatting_func=formatting_prompts_func,
            processing_class=tokenizer,
            args=training_args,
        )
        trainer.train()
        print("Saving the LoRA Adapter...")
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("Done!")
    except Exception as e:
        import traceback
        with open("crash_log.txt", "w") as f:
            f.write(traceback.format_exc())
        raise e

if __name__ == "__main__":
    main()
