"""
baseline_evaluate.py — SOTA SSL / KD baselines for fair comparison to Label-Boost.

Four method families (Arm A = no large teacher; Arm B = same frozen teacher):

  FixMatch_hard / FixMatch_soft   (A)  student self-PL, weak→strong, confidence τ
  MeanTeacher                     (A)  EMA teacher, MSE consistency
  MetaPseudoLabels                (A)  co-trained ResNet18 teacher (soft MPL)
  KD_ViTMAE_T4 / KD_MII_T4        (B)  classical Hinton KD (T=4) with Label-Boost teachers

Fair-comparison protocol (matched to evaluate.py / Label-Boost):
  - Same ResNet18 student (from scratch), AdamW lr=1e-3 / wd=1e-4, 300 epochs
  - Same patient-level 5-fold splits, annotation-ratio grid, val-CE freeze, PHAIR JSON
  - FixMatch uses the *same* model for pseudo-labels (canonical); no ViTMAE/MII graft
  - MPL teacher is same-capacity ResNet18, adaptive — not the frozen large teacher
  - KD reuses the same ViTMAE/MII probe checkpoints as ResNet18_{teacher}_l1.0
  - Strong aug is medical-safe (brightness/contrast/blur/±5° rot; no HFlip)
  - MPL omits UDA teacher auxiliary loss (controlled simplification)
  - λ_u / KD λ ramped with alpha_schedule over 100 epochs (cap 1.0)

Hyperparameters:
  FixMatch τ=0.95; MeanTeacher EMA decay=0.99; KD T=4; MPL soft PL + labeled finetune
  on the final 10% of epochs at lr=1e-5.

Usage:
    python baseline_evaluate.py --dataset hip --experiment FixMatch_hard --ar 0.5 --gpu 0
    python baseline_evaluate.py --dataset hip --experiment KD_MII_T4 --ar 0.1,0.5,1.0 \\
        --probe-root eval_output --gpu 0
    python baseline_evaluate.py --dataset covid --gpu 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from evaluate import (
    DATASET_PRESETS,
    DEFAULT_EPOCHS,
    ENCODER_LR,
    ENCODER_WD,
    K_FOLDS,
    LAMBDA_RAMP_EPOCHS,
    NUM_CLASSES,
    NUM_WORKERS,
    STUDENT_LR,
    STUDENT_WD,
    TRAIN_RATIO,
    VALID_RATIO,
    DATA_SEED,
    MODEL_SEED,
    DatasetPreset,
    configure_deterministic,
    default_device,
    evaluate_test,
    load_teacher,
    parse_ar_list,
    resolve_output_roots,
    seed_everything,
    validate,
)
from lib.baselines.augment import StrongAugment
from lib.baselines.ema import EMATeacher
from lib.baselines.trainers import (
    train_epoch_fixmatch,
    train_epoch_kd,
    train_epoch_mean_teacher,
    train_epoch_mpl,
    train_epoch_supervised_finetune,
)
from lib.criterion import ClsCriterion
from lib.dataloader import create_dataloaders
from lib.utils.model_utils import alpha_schedule, save_checkpoint
from models.resnet import CustomResNet
from phair_eval_suite.ldh_eval import evaluate_cross_validation

# ---------------------------------------------------------------------------
# Baseline-specific constants
# ---------------------------------------------------------------------------
FIXMATCH_TAU = 0.95
EMA_DECAY = 0.99
KD_TEMPERATURE = 4.0
MPL_FINETUNE_FRACTION = 0.10
MPL_FINETUNE_LR = 1e-5
UNSUP_LAM_MAX = 1.0


@dataclass
class Experiment:
    name: str
    method: str  # fixmatch | mean_teacher | mpl | kd
    teacher: Optional[str] = None  # ViTMAE / MII for KD only
    pl_soft: bool = False  # FixMatch soft vs hard PL
    tau: float = FIXMATCH_TAU
    temperature: float = KD_TEMPERATURE
    ema_decay: float = EMA_DECAY
    # Experiment directory used to locate teacher probes (KD fair reuse)
    probe_experiment: Optional[str] = None


EXPERIMENTS: dict[str, Experiment] = {
    "FixMatch_hard": Experiment("FixMatch_hard", method="fixmatch", pl_soft=False),
    "FixMatch_soft": Experiment("FixMatch_soft", method="fixmatch", pl_soft=True),
    "MeanTeacher": Experiment("MeanTeacher", method="mean_teacher"),
    "MetaPseudoLabels": Experiment("MetaPseudoLabels", method="mpl"),
    "KD_ViTMAE_T4": Experiment(
        "KD_ViTMAE_T4",
        method="kd",
        teacher="ViTMAE",
        temperature=KD_TEMPERATURE,
        probe_experiment="ResNet18_ViTMAE_l1.0",
    ),
    "KD_MII_T4": Experiment(
        "KD_MII_T4",
        method="kd",
        teacher="MII",
        temperature=KD_TEMPERATURE,
        probe_experiment="ResNet18_MII_l1.0",
    ),
}


def _make_student(preset: DatasetPreset, device: torch.device) -> CustomResNet:
    return CustomResNet(
        num_channels=preset.num_output_channels,
        num_classes=NUM_CLASSES,
        pretrained=False,
        device=device,
    )


def run_fold(
    preset: DatasetPreset,
    experiment: Experiment,
    ar: float,
    k: int,
    device: torch.device,
    epochs: int,
    eval_root: str,
    output_root: str,
    num_workers: int = NUM_WORKERS,
    log_curves: bool = True,
    probe_root: Optional[str] = None,
) -> dict:
    """Train one fold for a baseline method; freeze at best val CE; test once."""
    ar_dir = Path(eval_root) / preset.dataset_name / experiment.name / f"ar{ar}"
    ar_dir.mkdir(parents=True, exist_ok=True)
    teacher_eval_root = probe_root if probe_root is not None else eval_root

    ssl = ar != 1.0
    dataset_path = f"./datasets/{preset.dataset_name}"
    seed_everything(DATA_SEED)

    loaders = create_dataloaders(
        dataset_name=preset.dataset_name,
        dataset_path=dataset_path,
        class_sizes=list(preset.class_sizes),
        seed=DATA_SEED,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VALID_RATIO,
        annot_ratio=ar,
        batch_size=preset.batch_size,
        num_workers=num_workers,
        num_output_channels=preset.num_output_channels,
        image_size=preset.image_size,
        k_folds=K_FOLDS,
        k_fold_index=k,
    )
    if ssl:
        dloader_u, dloader_l, valid_dloader, test_dloader = loaders
    else:
        dloader_l, valid_dloader, test_dloader = loaders
        dloader_u = None

    print(
        f"  [k={k}] method={experiment.method}"
        f"  train_l={len(dloader_l.dataset)}"
        f"  train_u={len(dloader_u.dataset) if dloader_u else 0}"
        f"  valid={len(valid_dloader.dataset)} test={len(test_dloader.dataset)}"
    )

    # Frozen large teacher (KD only)
    frozen_teacher = None
    pseudolabel_acc = None
    if experiment.method == "kd":
        assert experiment.teacher is not None
        probe_exp = experiment.probe_experiment or experiment.name
        frozen_teacher = load_teacher(
            experiment.teacher,
            preset.dataset_name,
            probe_exp,
            k,
            ar,
            device,
            eval_root=teacher_eval_root,
            output_root=output_root,
            dataset_path=dataset_path,
            class_sizes=list(preset.class_sizes),
            softlabels=False,
            image_size=preset.image_size,
            num_output_channels=preset.num_output_channels,
            num_workers=num_workers,
        )
        print(f"  [k={k}] Loaded frozen teacher {experiment.teacher}"
              f"  (probes from experiment={probe_exp})")

    seed_everything(MODEL_SEED)
    model = _make_student(preset, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=STUDENT_LR, weight_decay=STUDENT_WD,
    )
    criterion = ClsCriterion()
    strong_aug = StrongAugment()

    ema: Optional[EMATeacher] = None
    mpl_teacher: Optional[CustomResNet] = None
    opt_t: Optional[torch.optim.Optimizer] = None
    if experiment.method == "mean_teacher":
        ema = EMATeacher(model, decay=experiment.ema_decay).to(device)
    elif experiment.method == "mpl":
        mpl_teacher = _make_student(preset, device)
        opt_t = torch.optim.AdamW(
            mpl_teacher.parameters(), lr=STUDENT_LR, weight_decay=STUDENT_WD,
        )

    log_dir = ar_dir / f"k{k}"
    logger = SummaryWriter(log_dir=str(log_dir))

    best_val_loss = math.inf
    best_epoch = -1
    mpl_finetune_start = int(epochs * (1.0 - MPL_FINETUNE_FRACTION))

    for epoch in range(epochs):
        lam = alpha_schedule(0, epoch, LAMBDA_RAMP_EPOCHS, UNSUP_LAM_MAX)

        if experiment.method == "fixmatch":
            train_stats = train_epoch_fixmatch(
                model, optimizer, dloader_l, dloader_u,
                lam_u=lam, tau=experiment.tau, pl_soft=experiment.pl_soft,
                num_classes=NUM_CLASSES, device=device, strong_aug=strong_aug,
            )
        elif experiment.method == "mean_teacher":
            assert ema is not None
            train_stats = train_epoch_mean_teacher(
                model, ema, optimizer, dloader_l, dloader_u,
                lam_u=lam, num_classes=NUM_CLASSES, device=device,
                strong_aug=strong_aug,
            )
        elif experiment.method == "kd":
            assert frozen_teacher is not None
            train_stats = train_epoch_kd(
                model, frozen_teacher, optimizer, dloader_l, dloader_u,
                lam=lam, temperature=experiment.temperature,
                num_classes=NUM_CLASSES, device=device,
            )
        elif experiment.method == "mpl":
            assert mpl_teacher is not None and opt_t is not None
            if epoch >= mpl_finetune_start:
                # Paper-style labeled finetune of the student
                if epoch == mpl_finetune_start:
                    for g in optimizer.param_groups:
                        g["lr"] = MPL_FINETUNE_LR
                    print(f"  [k={k}] MPL finetune phase starting "
                          f"(lr={MPL_FINETUNE_LR})")
                train_stats = train_epoch_supervised_finetune(
                    model, optimizer, dloader_l, NUM_CLASSES, device,
                )
            else:
                train_stats = train_epoch_mpl(
                    model, mpl_teacher, optimizer, opt_t,
                    dloader_l, dloader_u, NUM_CLASSES, device,
                    student_lr=STUDENT_LR,
                )
        else:
            raise ValueError(f"Unknown method: {experiment.method}")

        train_ce = train_stats["hard_ce"]
        val_out = validate(
            model, criterion, valid_dloader, softlabels=False, device=device,
            with_metrics=log_curves,
        )
        val_loss = val_out["ce"] if log_curves else val_out

        logger.add_scalar("Train/CE Loss", train_ce, epoch + 1)
        logger.add_scalar("Valid/CE Losses", val_loss, epoch + 1)
        logger.add_scalar("Train/Lambda", lam, epoch + 1)
        if log_curves:
            logger.add_scalar("Valid/Top 1 accuracy", val_out["top1"], epoch + 1)
            logger.add_scalar("Valid/AUROC", val_out["auroc"], epoch + 1)
            for key in ("mask_rate", "unsup_loss", "consistency", "kd_loss", "meta_loss"):
                if train_stats.get(key) is not None:
                    logger.add_scalar(f"Train/{key}", train_stats[key], epoch + 1)

        if math.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(str(ar_dir), {
                "epoch": epoch + 1,
                "best_loss": best_val_loss,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "method": experiment.method,
            }, filename=f"k{k}_best")

        if (epoch + 1) % 50 == 0 or epoch == 0:
            val_str = f"{val_loss:.4f}" if math.isfinite(val_loss) else str(val_loss)
            train_str = f"{train_ce:.4f}" if math.isfinite(train_ce) else str(train_ce)
            extra = ""
            if log_curves:
                extra = f"  val_top1={val_out['top1']:.4f}  val_auroc={val_out['auroc']:.4f}"
            print(
                f"  [k={k}] epoch {epoch+1:>3}/{epochs}"
                f"  train_ce={train_str}  val_ce={val_str}"
                f"  best@{best_epoch+1}{extra}"
            )

    ckpt_path = ar_dir / "checkpoints" / f"k{k}_best.pth.tar"
    if not ckpt_path.is_file():
        raise RuntimeError(
            f"No best checkpoint was written to {ckpt_path}. "
            "Validation loss was never a finite improvement over the previous best."
        )
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(
        f"  [k={k}] Loaded best checkpoint (epoch {ckpt['epoch']},"
        f" val_loss={ckpt['best_loss']:.4f})"
    )

    results = evaluate_test(model, test_dloader, device)
    logger.add_scalar("Test/Top 1 accuracy", results["top1"], 1)
    logger.add_scalar("Test/AUROC", results["auroc"], 1)
    if log_curves:
        logger.add_scalar("Train/Best Epoch", best_epoch + 1, 1)
    logger.flush()
    logger.close()

    with open(ar_dir / f"fold_{k}_predictions.json", "w") as f:
        json.dump({
            "y_true": results["y_true"].tolist(),
            "y_proba": results["y_prob"].tolist(),
        }, f)

    results["pseudolabel_acc"] = pseudolabel_acc
    print(f"  [k={k}] top1={results['top1']:.4f}  auroc={results['auroc']:.4f}")
    return results


def run_experiment(
    preset: DatasetPreset,
    experiment: Experiment,
    ar_list: list[float],
    device: torch.device,
    epochs: int = DEFAULT_EPOCHS,
    eval_root: str = "baseline_output",
    output_root: str = "output",
    num_workers: int = NUM_WORKERS,
    log_curves: bool = True,
    probe_root: Optional[str] = None,
):
    print(f"\n{'='*60}")
    print(f"Baseline: {experiment.name}  |  Dataset: {preset.dataset_name}")
    print(f"Method: {experiment.method}  |  Teacher: {experiment.teacher or 'None'}")
    print(f"Epochs: {epochs}  |  Device: {device}  |  eval_root: {eval_root}")
    if probe_root is not None and probe_root != eval_root:
        print(f"Probe root: {probe_root}")
    print(f"{'='*60}")

    cross_ar_rows: list[dict] = []

    for ar in ar_list:
        print(f"\n--- ar={ar} ---")
        fold_results: list[dict] = []

        for k in range(K_FOLDS):
            res = run_fold(
                preset, experiment, ar, k, device, epochs, eval_root, output_root,
                num_workers=num_workers,
                log_curves=log_curves,
                probe_root=probe_root,
            )
            fold_results.append({
                "k": k, "top1": res["top1"], "auroc": res["auroc"],
                "pseudolabel_acc": res["pseudolabel_acc"],
            })

        top1s = [r["top1"] for r in fold_results]
        aurocs = [r["auroc"] for r in fold_results]
        summary_mean = {"k": "mean", "top1": np.mean(top1s), "auroc": np.mean(aurocs)}
        summary_std = {
            "k": "std",
            "top1": np.std(top1s) / np.sqrt(K_FOLDS),
            "auroc": np.std(aurocs) / np.sqrt(K_FOLDS),
        }

        ar_dir = Path(eval_root) / preset.dataset_name / experiment.name / f"ar{ar}"

        print(f"  Running PHAIR evaluation for ar={ar}...")
        try:
            evaluate_cross_validation(str(ar_dir), recalibrate=False, threshold=None)
        except Exception as e:
            print(f"  PHAIR evaluation failed: {e}")

        for k in range(K_FOLDS):
            mj = ar_dir / f"fold_{k}" / "metrics.json"
            if mj.is_file():
                with open(mj, encoding="utf-8") as f:
                    m = json.load(f)
                fold_results[k]["brier_score"] = float(m["brier_score"])
                fold_results[k]["calibration_slope"] = float(m["calibration_slope"])
            else:
                fold_results[k]["brier_score"] = float("nan")
                fold_results[k]["calibration_slope"] = float("nan")

        briers = np.array(
            [fold_results[j]["brier_score"] for j in range(K_FOLDS)], dtype=float
        )
        slopes = np.array(
            [fold_results[j]["calibration_slope"] for j in range(K_FOLDS)],
            dtype=float,
        )
        if np.all(np.isfinite(briers)):
            summary_mean["brier_score"] = float(np.mean(briers))
            summary_std["brier_score"] = float(np.std(briers) / np.sqrt(K_FOLDS))
        if np.all(np.isfinite(slopes)):
            summary_mean["calibration_slope"] = float(np.mean(slopes))
            summary_std["calibration_slope"] = float(
                np.std(slopes) / np.sqrt(K_FOLDS)
            )

        fold_results.append(summary_mean)
        fold_results.append(summary_std)
        pd.DataFrame(fold_results).to_csv(ar_dir / "summary.csv", index=False)

        row_out: dict = {
            "ar": ar,
            "top1_mean": np.mean(top1s),
            "top1_stderr": np.std(top1s) / np.sqrt(K_FOLDS),
            "auroc_mean": np.mean(aurocs),
            "auroc_stderr": np.std(aurocs) / np.sqrt(K_FOLDS),
        }
        if np.all(np.isfinite(briers)):
            row_out["brier_score_mean"] = float(np.mean(briers))
            row_out["brier_score_stderr"] = float(np.std(briers) / np.sqrt(K_FOLDS))
        if np.all(np.isfinite(slopes)):
            row_out["calibration_slope_mean"] = float(np.mean(slopes))
            row_out["calibration_slope_stderr"] = float(
                np.std(slopes) / np.sqrt(K_FOLDS)
            )
        cross_ar_rows.append(row_out)

        print(
            f"  ar={ar} complete:"
            f" top1={np.mean(top1s):.4f} +/- {np.std(top1s)/np.sqrt(K_FOLDS):.4f}"
            f" auroc={np.mean(aurocs):.4f} +/- {np.std(aurocs)/np.sqrt(K_FOLDS):.4f}"
        )

    exp_dir = Path(eval_root) / preset.dataset_name / experiment.name
    summary_path = exp_dir / "summary_all_ar.csv"
    pd.DataFrame(cross_ar_rows).to_csv(summary_path, index=False)
    print(f"\nSaved cross-AR summary: {summary_path}")


def main_cli():
    parser = argparse.ArgumentParser(
        description="Train SOTA SSL/KD baselines for fair comparison to Label-Boost.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python baseline_evaluate.py --dataset hip --experiment FixMatch_hard --ar 0.5 --gpu 0
  python baseline_evaluate.py --dataset hip --experiment MeanTeacher --ar 0.1,0.5,1.0 --gpu 0
  python baseline_evaluate.py --dataset hip --experiment MetaPseudoLabels --ar 0.5 --gpu 0
  python baseline_evaluate.py --dataset hip --experiment KD_MII_T4 --ar 0.5 \\
      --probe-root eval_output --gpu 0
  python baseline_evaluate.py --dataset covid --gpu 0
""",
    )
    parser.add_argument(
        "--dataset", required=True, choices=list(DATASET_PRESETS.keys()),
        help="Dataset preset",
    )
    parser.add_argument(
        "--experiment", default=None, choices=list(EXPERIMENTS.keys()),
        help="Baseline experiment name (omit to run all)",
    )
    parser.add_argument(
        "--ar", type=str, default=None,
        metavar="RATIO[,RATIO...]",
        help="Annotation ratio(s); omit for full sweep 0.1-1.0",
    )
    parser.add_argument(
        "--eval-root", default="baseline_output",
        help="Output root (default: baseline_output). "
             "With log-curves (default) this becomes curves_{eval-root}.",
    )
    parser.add_argument(
        "--output-root", default="output",
        help="Root for ViT-MAE encoder checkpoints (KD only)",
    )
    parser.add_argument(
        "--log-curves", dest="log_curves", action="store_true", default=True,
        help="Log per-epoch curves under curves_{eval-root}/ (default: on)",
    )
    parser.add_argument(
        "--no-log-curves", dest="log_curves", action="store_false",
        help="Disable curve logging; write to unprefixed --eval-root",
    )
    parser.add_argument(
        "--probe-root", default="eval_output",
        help="Root for frozen-teacher linear probes (KD). Default: eval_output "
             "so Label-Boost probes are reused via probe_experiment mapping.",
    )
    parser.add_argument("--device", default=None, help="Device override")
    parser.add_argument(
        "--gpu", default=None, type=str,
        help="CUDA_VISIBLE_DEVICES value, e.g. '0' or '0,1'",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--num-workers", type=int, default=NUM_WORKERS,
        help=f"DataLoader workers (default: {NUM_WORKERS})",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    configure_deterministic()

    preset = DATASET_PRESETS[args.dataset]
    device = torch.device(args.device) if args.device else default_device()
    if args.ar is not None:
        try:
            ar_list = parse_ar_list(args.ar)
        except argparse.ArgumentTypeError as e:
            parser.error(str(e))
    else:
        ar_list = [round(i * 0.1, 1) for i in range(1, 11)]

    if args.experiment:
        experiments = [EXPERIMENTS[args.experiment]]
    else:
        experiments = list(EXPERIMENTS.values())

    eval_root, probe_root = resolve_output_roots(
        args.eval_root, args.log_curves, args.probe_root,
    )

    print(f"Device: {device}")
    print(f"Experiments: {[e.name for e in experiments]}")
    print(f"Annotation ratios: {ar_list}")
    print(f"eval_root: {eval_root}  probe_root: {probe_root}  log_curves: {args.log_curves}")

    for exp in experiments:
        run_experiment(
            preset, exp, ar_list, device,
            epochs=args.epochs,
            eval_root=eval_root,
            output_root=args.output_root,
            num_workers=args.num_workers,
            log_curves=args.log_curves,
            probe_root=probe_root,
        )


if __name__ == "__main__":
    main_cli()
