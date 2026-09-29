#!/bin/bash
# YS-08 / YS-10 bounded real-data runs (2026-09-28). Re-runnable.
set -e
cd /root/geo-yield-prediction
export YIELDSAT_SOURCE_ROOT=/home1/pupil/SMATousi/YieldSAT/Preprocessed YIELDSAT_ARTIFACT_ROOT=/root/yieldsat_artifacts
PY=.conda/phase0/bin/python
ALL="Argentina Brazil Germany Uruguay"
R=/root/yieldsat_artifacts/runs
COMMON="--data_contract yieldsat_preprocessed_v1 --epochs 20 --steps_per_epoch 500 --batch_size 512 --num_workers 6"
for c in Brazil Argentina; do $PY yieldsat_prepare.py semantics --countries $c > $R/semantics_$c.log 2>&1; done
$PY yieldsat_prepare.py snapshot --countries $ALL > $R/snapshot.log 2>&1 || echo "snapshot check failed (see log)"
# splits were created with seed 0 before training (pooled_farm_s0, loco_uruguay_s0, pooled_block20_s0, ...)
$PY main_yieldsat_finetune.py $COMMON --countries $ALL --split pooled_farm_s0 --output_dir $R/pooled_farm > $R/pooled_farm.log 2>&1 &
$PY main_yieldsat_finetune.py $COMMON --countries $ALL --split loco_uruguay_s0 --output_dir $R/loco_uruguay > $R/loco_uruguay.log 2>&1 &
$PY main_yieldsat_finetune.py --data_contract yieldsat_preprocessed_v1 --epochs 15 --steps_per_epoch 300 --batch_size 512 --countries Germany --split germany_farm_s0 --output_dir $R/pilot_germany_farm > $R/pilot_germany_farm.log 2>&1 &
wait
$PY main_yieldsat_finetune.py $COMMON --countries $ALL --split pooled_block20_s0 --output_dir $R/pooled_block20 > $R/pooled_block20.log 2>&1 &
# YS-10: yield-free pretraining on the pooled training partition only
$PY main_yieldsat_finetune.py $COMMON --mode pretrain --countries $ALL --split pooled_farm_s0 --output_dir $R/pretrain_pooled > $R/pretrain_pooled.log 2>&1 &
wait
PT=$R/pretrain_pooled/sensor_checkpoint.pth
for frac in 0.1 1.0; do
  $PY main_yieldsat_finetune.py $COMMON --countries Uruguay --split pooled_farm_s0 --train_fraction $frac --output_dir $R/uruguay_scratch_f$frac > $R/uruguay_scratch_f$frac.log 2>&1 &
  $PY main_yieldsat_finetune.py $COMMON --countries Uruguay --split pooled_farm_s0 --train_fraction $frac --init_sensor_ckpt $PT --output_dir $R/uruguay_pretrained_f$frac > $R/uruguay_pretrained_f$frac.log 2>&1 &
  wait
done
echo ALL DONE
