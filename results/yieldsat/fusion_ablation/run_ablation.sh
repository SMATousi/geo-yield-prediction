#!/bin/bash
# YS-11 fusion ablation (2026-09-29). Code frozen at commit b34f1cd (git worktree).
cd /root/ys11_ablation_src
export YIELDSAT_SOURCE_ROOT=/home1/pupil/SMATousi/YieldSAT/Preprocessed YIELDSAT_ARTIFACT_ROOT=/root/yieldsat_artifacts
PY=/root/geo-yield-prediction/.conda/phase0/bin/python
OUT=/root/yieldsat_artifacts/runs/fusion_ablation
ALL="Argentina Brazil Germany Uruguay"
COMMON="--data_contract yieldsat_preprocessed_v1 --countries $ALL --epochs 20 --steps_per_epoch 500 --batch_size 512 --num_workers 5 --eval_drop_stream yieldsat_s2 --eval_drop_frac 0.5 --save_maps 0"
jobs_list=()
for split in pooled_farm_s0 pooled_block20_s0; do
  for seed in 0 1 2; do
    for fusion in perceiver_summary perceiver_tokens concat_mlp token_transformer; do
      extra=""; [ $fusion = perceiver_tokens ] && extra="--num_latents 16"
      jobs_list+=("$split|$seed|$fusion|$extra|${split}_${fusion}_seed$seed")
    done
  done
done
# latent-count sweep for perceiver_tokens (pooled farm split, seed 0)
for nl in 8 32; do jobs_list+=("pooled_farm_s0|0|perceiver_tokens|--num_latents $nl|pooled_farm_s0_perceiver_tokens_L${nl}_seed0"); done
run() { IFS='|' read split seed fusion extra name <<< "$1"
  [ -f $OUT/$name/report.json ] && return
  $PY main_yieldsat_finetune.py $COMMON --split $split --seed $seed --fusion $fusion $extra --output_dir $OUT/$name > $OUT/$name.log 2>&1; }
n=0
for j in "${jobs_list[@]}"; do run "$j" & n=$((n+1)); if [ $((n % 3)) -eq 0 ]; then wait; fi; done
wait
echo ABLATION DONE
