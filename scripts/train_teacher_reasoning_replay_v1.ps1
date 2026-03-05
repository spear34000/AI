param(
    [string]$Checkpoint = "artifacts_bigslm_50m_safe_fromscratch_v1_runC_stage2/slm_last.pt",
    [string]$DataPath = "data/teacher_reasoning_replay_mix_v1.jsonl",
    [string]$OutputDir = "artifacts_teacher_reasoning_replay_v1_runA",
    [int]$AdditionalSteps = 1800,
    [double]$Lr = 1.5e-5,
    [int]$EvalInterval = 120,
    [int]$SaveInterval = 600,
    [int]$SeqLen = 512,
    [int]$LoraR = 16,
    [double]$LoraAlpha = 32.0,
    [double]$LoraDropout = 0.03,
    [string]$LoraTargets = "attn.qkv,attn.proj,mlp.0,mlp.2"
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
  --weight_decay 0.01 `
  --warmup_steps 80 `
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
