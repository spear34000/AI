@echo off
setlocal
cd /d C:\Users\User\Desktop\test_ai

set "PY=C:\Program Files\Python311\python.exe"
set "TOKENIZER=artifacts_tokenizer_spm_serious_v1\serious_ko_spm16k_v1.model"
set "CORPUS=data\final_datasets_corpus_v1.jsonl"
set "DEFS=data\final_datasets_defs_boost_v1.jsonl"
set "TOOLCACHE=data\tool_knowledge_cache_final_v1.jsonl"

set "PHASE1_OUT=artifacts_final_continue_v1_corpus"
set "PHASE2_OUT=artifacts_final_continue_v1_defsboost"

set "PHASE1_RESUME=artifacts_serious_slm_v2_defsboost\slm_best.pt"
if exist "%PHASE1_OUT%\slm_best.pt" set "PHASE1_RESUME=%PHASE1_OUT%\slm_best.pt"

if exist "%PHASE2_OUT%\slm_best.pt" goto phase2

echo ==> Phase 1: continue on final corpus (resume=%PHASE1_RESUME%)
"%PY%" scripts\train_slm.py ^
  --data_path "%CORPUS%" ^
  --output_dir "%PHASE1_OUT%" ^
  --resume_from "%PHASE1_RESUME%" ^
  --resume_weights_only ^
  --reset_best_on_resume ^
  --tokenizer_type spm ^
  --tokenizer_model "%TOKENIZER%" ^
  --steps 2200 ^
  --batch_size 1 ^
  --grad_accum_steps 16 ^
  --seq_len 512 ^
  --lr 3e-5 ^
  --weight_decay 0.1 ^
  --warmup_steps 80 ^
  --grad_clip 1.0 ^
  --val_ratio 0.02 ^
  --ema_decay 0.0 ^
  --consistency_lambda 0.01 ^
  --consistency_temp 1.5 ^
  --consistency_warmup 400 ^
  --eval_interval 100 ^
  --save_interval 200 ^
  --eval_batches 8 ^
  --num_workers 0 ^
  --activation_checkpointing ^
  --skip_sample
if errorlevel 1 exit /b %errorlevel%

:phase2
set "PHASE2_RESUME=%PHASE1_OUT%\slm_best.pt"
if exist "%PHASE2_OUT%\slm_best.pt" set "PHASE2_RESUME=%PHASE2_OUT%\slm_best.pt"

echo ==> Phase 2: continue on final defs boost (resume=%PHASE2_RESUME%)
"%PY%" scripts\train_slm.py ^
  --data_path "%DEFS%" ^
  --output_dir "%PHASE2_OUT%" ^
  --resume_from "%PHASE2_RESUME%" ^
  --resume_weights_only ^
  --reset_best_on_resume ^
  --tokenizer_type spm ^
  --tokenizer_model "%TOKENIZER%" ^
  --steps 1200 ^
  --batch_size 1 ^
  --grad_accum_steps 16 ^
  --seq_len 512 ^
  --lr 2e-5 ^
  --weight_decay 0.1 ^
  --warmup_steps 50 ^
  --grad_clip 1.0 ^
  --val_ratio 0.02 ^
  --ema_decay 0.0 ^
  --consistency_lambda 0.0 ^
  --consistency_temp 1.5 ^
  --consistency_warmup 1000000 ^
  --eval_interval 50 ^
  --save_interval 100 ^
  --eval_batches 8 ^
  --num_workers 0 ^
  --plain_sft ^
  --activation_checkpointing ^
  --skip_sample
if errorlevel 1 exit /b %errorlevel%

echo ==> Eval strict with final tool cache
"%PY%" scripts\eval_ko_quality5x.py ^
  --candidate_checkpoint "%PHASE2_OUT%\slm_best.pt" ^
  --device cuda ^
  --tool_cache_path "%TOOLCACHE%" ^
  --out_json "%PHASE2_OUT%\eval_ko_quality5x_final.json"
exit /b %errorlevel%
