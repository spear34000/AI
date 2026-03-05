param(
    [string]$Checkpoint = "artifacts_bigslm_50m_safe_fromscratch_v1_runC_stage2/slm_last.pt",
    [string]$DataPath = "data/numeric_symbolic_bootstrap_v1_train.jsonl",
    [string]$OutputDir = "artifacts_numeric_symbolic_lora_v1_runA",
    [int]$AdditionalSteps = 1600,
    [double]$Lr = 2.0e-4,
    [int]$EvalInterval = 80,
    [int]$SaveInterval = 160,
    [int]$SeqLen = 128,
    [int]$LoraR = 32,
    [double]$LoraAlpha = 64.0,
    [double]$LoraDropout = 0.0,
    [string]$LoraTargets = "attn.qkv,attn.proj,mlp.0,mlp.2,lm_head"
)

$py = "C:\Program Files\Python311\python.exe"
$resumeStep = [int](& $py -c "import sys, torch; ck=torch.load(sys.argv[1], map_location='cpu', weights_only=False); print(int(ck.get('step', 0)))" $Checkpoint)
$targetSteps = $resumeStep + $AdditionalSteps

& $py -u scripts/train_slm.py `
  --data_path $DataPath `
  --output_dir $OutputDir `
  --steps $targetSteps `
  --batch_size 1 `
  --grad_accum_steps 16 `
  --seq_len $SeqLen `
  --lr $Lr `
  --weight_decay 0.0 `
  --warmup_steps 40 `
  --grad_clip 1.0 `
  --val_ratio 0.02 `
  --ema_decay 0.0 `
  --consistency_lambda 0.0 `
  --consistency_temp 1.5 `
  --consistency_warmup 1000000 `
  --eval_interval $EvalInterval `
  --save_interval $SaveInterval `
  --eval_batches 16 `
  --num_workers 0 `
  --resume_from $Checkpoint `
  --resume_weights_only `
  --reset_best_on_resume `
  --plain_sft `
  --activation_checkpointing `
  --skip_sample `
  --lora_r $LoraR `
  --lora_alpha $LoraAlpha `
  --lora_dropout $LoraDropout `
  --lora_targets $LoraTargets
