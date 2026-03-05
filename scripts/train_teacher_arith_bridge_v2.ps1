param(
    [string]$ResumeFrom = "artifacts_teacher_muldiv_curriculum_v1_runA/slm_best.pt",
    [string]$OutputDir = "artifacts_teacher_arith_bridge_v2_runA"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = "C:\Program Files\Python311\python.exe"
if (-not (Test-Path $py)) {
    throw "Python not found: $py"
}

& $py scripts/build_teacher_arith_bridge_v2.py
if ($LASTEXITCODE -ne 0) {
    throw "dataset build failed"
}

& $py -u scripts/train_slm.py `
    --data_path data/teacher_arith_bridge_v2_train.jsonl `
    --output_dir $OutputDir `
    --steps 11800 `
    --batch_size 1 `
    --grad_accum_steps 16 `
    --seq_len 384 `
    --lr 6.0e-6 `
    --weight_decay 0.1 `
    --warmup_steps 120 `
    --grad_clip 1.0 `
    --val_ratio 0.02 `
    --eval_interval 200 `
    --save_interval 400 `
    --eval_batches 24 `
    --num_workers 0 `
    --d_model 576 `
    --n_heads 9 `
    --n_layers 12 `
    --mlp_mult 4 `
    --tokenizer_type spm `
    --tokenizer_model artifacts_tokenizer_spm_from_stage2_v1/spm16k.model `
    --activation_checkpointing `
    --skip_sample `
    --turbo_mode `
    --turbo_min_seq_len 256 `
    --turbo_seq_warmup_ratio 0.35 `
    --turbo_max_frozen_layers 4 `
    --turbo_depth_warmup_ratio 0.25 `
    --ema_decay 0.0 `
    --consistency_lambda 0.0 `
    --plain_sft `
    --resume_from $ResumeFrom `
    --resume_weights_only `
    --reset_best_on_resume

if ($LASTEXITCODE -ne 0) {
    throw "training failed"
}
