#!/usr/bin/env bash
# Label-boost (new_evaluate.py, curves_eval_output), one step, label-type pool:
#   CE_hard(labeled) + λ * teacher_loss averaged over labeled and unlabeled.
#   T=1 soft cross-entropy. T=4 Hinton T^2 KL on that teacher term.
#
# KD two-step (baseline_evaluate.py, curves_baseline_output). Two updates.
#   label:  hard step, then one KD step averaged over labeled and unlabeled
#   sample: hard+KD on the labeled batch, then KD on the unlabeled batch
#
# Already trained, and left commented out:
#   label,  T=1  same update as curves_eval_output/*/ResNet18_*_l1.0
#   sample, T=4  same update as curves_baseline_output/*/KD_*_T4
# Launched below:
#   label,  T=4  KD_*_T4_2step
#   sample, T=1  KD_*_T1_2step_sample
#
# Defaults: 300 epochs, AdamW lr=1e-3 / wd=1e-4, lambda ramped to 1.0 over
# 100 epochs, five folds, every annotation ratio from 0.1 to 1.0.
# Hip uses ViTMAE and MII. Covid uses MII. Probes come from eval_output.
#
# Usage, from the repo root:
#   bash configs/run_signal_diff.sh 0

set -euo pipefail

GPU="${1:-0}"
cd "$(dirname "$0")/.."

# --- Label-boost, one step, T=1 ---
python new_evaluate.py --dataset hip --experiment ResNet18_ViTMAE_l1.0_1step \
    --gpu "$GPU" --probe-root eval_output
python new_evaluate.py --dataset hip --experiment ResNet18_MII_l1.0_1step \
    --gpu "$GPU" --probe-root eval_output
python new_evaluate.py --dataset covid --experiment ResNet18_MII_l1.0_1step \
    --gpu "$GPU" --probe-root eval_output

# --- Label-boost, one step, T=4 ---
python new_evaluate.py --dataset hip --experiment ResNet18_ViTMAE_l1.0_1step_T4 \
    --gpu "$GPU" --probe-root eval_output
python new_evaluate.py --dataset hip --experiment ResNet18_MII_l1.0_1step_T4 \
    --gpu "$GPU" --probe-root eval_output
python new_evaluate.py --dataset covid --experiment ResNet18_MII_l1.0_1step_T4 \
    --gpu "$GPU" --probe-root eval_output

# --- KD two-step, label, T=1. Same update as ResNet18_*_l1.0. ---
# python baseline_evaluate.py --dataset hip --experiment KD_ViTMAE_T1_2step \
#     --gpu "$GPU" --probe-root eval_output
# python baseline_evaluate.py --dataset hip --experiment KD_MII_T1_2step \
#     --gpu "$GPU" --probe-root eval_output
# python baseline_evaluate.py --dataset covid --experiment KD_MII_T1_2step \
#     --gpu "$GPU" --probe-root eval_output

# --- KD two-step, label, T=4 ---
python baseline_evaluate.py --dataset hip --experiment KD_ViTMAE_T4_2step \
    --gpu "$GPU" --probe-root eval_output
python baseline_evaluate.py --dataset hip --experiment KD_MII_T4_2step \
    --gpu "$GPU" --probe-root eval_output
python baseline_evaluate.py --dataset covid --experiment KD_MII_T4_2step \
    --gpu "$GPU" --probe-root eval_output

# --- KD two-step, sample, T=1 ---
python baseline_evaluate.py --dataset hip --experiment KD_ViTMAE_T1_2step_sample \
    --gpu "$GPU" --probe-root eval_output
python baseline_evaluate.py --dataset hip --experiment KD_MII_T1_2step_sample \
    --gpu "$GPU" --probe-root eval_output
python baseline_evaluate.py --dataset covid --experiment KD_MII_T1_2step_sample \
    --gpu "$GPU" --probe-root eval_output

# --- KD two-step, sample, T=4. Same update as KD_*_T4. ---
# python baseline_evaluate.py --dataset hip --experiment KD_ViTMAE_T4_2step_sample \
#     --gpu "$GPU" --probe-root eval_output
# python baseline_evaluate.py --dataset hip --experiment KD_MII_T4_2step_sample \
#     --gpu "$GPU" --probe-root eval_output
# python baseline_evaluate.py --dataset covid --experiment KD_MII_T4_2step_sample \
#     --gpu "$GPU" --probe-root eval_output
