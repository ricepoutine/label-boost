"""
evaluate.py - Clean Training + Evaluation Pipeline

Trains student models (with optional teacher pseudolabels) using proper
train/validate/freeze/test protocol and generates publication-grade metrics
via phair_eval_suite.

Naming convention:
    {student}_{?teacher}_l{?lambda}
    e.g. ResNet18, ResNet18_soft, ResNet18_ViTMAE_l1.0, ResNet18_MII_l1.0

Usage:
    python evaluate.py --dataset hip --experiment ResNet18_ViTMAE_l1.0 --ar 1.0
    python evaluate.py --dataset hip --experiment ResNet18_MII_l1.0 --ar 0.5,0.6,0.7,0.8,0.9,1.0
    python evaluate.py --dataset hip --experiment ResNet18_ViTMAE_l1.0
    python evaluate.py --dataset hip
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from torch import tensor
from torchmetrics.classification import BinaryAUROC

from torch.utils.tensorboard import SummaryWriter
from lib.criterion import ClsCriterion, eps as CLS_EPS
from lib.dataloader import create_dataloaders
from lib.utils.model_utils import save_checkpoint, alpha_schedule
from models.resnet import CustomResNet
from models.vitmae import CustomViTMAE
from models.mii import CustomMII
from phair_eval_suite.ldh_eval import evaluate_cross_validation
from probe import ensure_teacher_linear_probe, resolve_teacher_paths

# ---------------------------------------------------------------------------
# Constants (match main.py defaults for hip experiments)
# ---------------------------------------------------------------------------
NUM_CLASSES = 2
LR = 1e-3
WD = 1e-4
NUM_WORKERS = 8
TRAIN_RATIO = 0.7
VALID_RATIO = 0.1
DATA_SEED = 0
MODEL_SEED = 0
K_FOLDS = 5
DEFAULT_EPOCHS = 300
LAMBDA_RAMP_EPOCHS = 100

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetPreset:
    dataset_name: str
    batch_size: int
    class_sizes: tuple[int, int] = (2000, 2000)
    image_size: tuple[int, int] = (224, 224)
    num_output_channels: int = 3


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "hip": DatasetPreset("hip734_cropped", batch_size=16),
    "covid": DatasetPreset("Covid", batch_size=64),
}


@dataclass
class Experiment:
    name: str
    teacher: Optional[str] = None
    lam: float = 1.0
    softlabels: bool = False
    ablation: bool = False


EXPERIMENTS: dict[str, Experiment] = {
    "ResNet18": Experiment("ResNet18"),
    "ResNet18_soft": Experiment("ResNet18_soft", softlabels=True),
    "ResNet18_ViTMAE_l0.5": Experiment("ResNet18_ViTMAE_l0.5", teacher="ViTMAE", lam=0.5),
    "ResNet18_ViTMAE_l1.0": Experiment("ResNet18_ViTMAE_l1.0", teacher="ViTMAE", lam=1.0),
    "ResNet18_ViTMAE_l1.5": Experiment("ResNet18_ViTMAE_l1.5", teacher="ViTMAE", lam=1.5),
    "ResNet18_ViTMAE_l1.0_ablation": Experiment("ResNet18_ViTMAE_l1.0_ablation", teacher="ViTMAE", lam=1.0, ablation=True),
    "ResNet18_MII_l0.5": Experiment("ResNet18_MII_l0.5", teacher="MII", lam=0.5),
    "ResNet18_MII_l1.0": Experiment("ResNet18_MII_l1.0", teacher="MII", lam=1.0),
    "ResNet18_MII_l1.5": Experiment("ResNet18_MII_l1.5", teacher="MII", lam=1.5),
    "ResNet18_MII_l1.0_ablation": Experiment("ResNet18_MII_l1.0_ablation", teacher="MII", lam=1.0, ablation=True),
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> torch.Generator:
    """Seed all RNGs and return a torch.Generator for DataLoader use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)           # seeds CPU + current CUDA device
    torch.cuda.manual_seed_all(seed)  # seeds *all* CUDA devices
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def configure_deterministic():
    """One-time call to force deterministic behaviour across backends."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Strongest guarantee: error on any non-deterministic op.
    # CUBLAS_WORKSPACE_CONFIG is required for CUDA >= 10.2.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ---------------------------------------------------------------------------
# Teacher: encoder in output/; linear probe in eval_output/.../checkpoints/ (see probe.py)
# ---------------------------------------------------------------------------


def load_teacher(
    teacher_name: str,
    dataset_name: str,
    experiment_name: str,
    k: int,
    ar: float,
    device: torch.device,
    eval_root: str = "eval_output",
    output_root: str = "output",
    *,
    dataset_path: Optional[str] = None,
    class_sizes: Optional[list[int]] = None,
    softlabels: bool = False,
    image_size: tuple[int, int] = (224, 224),
    num_output_channels: int = 3,
    num_workers: int = NUM_WORKERS,
) -> torch.nn.Module:
    if dataset_path is not None and class_sizes is not None:
        ensure_teacher_linear_probe(
            teacher_name=teacher_name,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            dataset_path=dataset_path,
            class_sizes=class_sizes,
            k=k,
            ar=ar,
            softlabels=softlabels,
            device=device,
            eval_root=eval_root,
            output_root=output_root,
            train_ratio=TRAIN_RATIO,
            valid_ratio=VALID_RATIO,
            data_seed=DATA_SEED,
            num_workers=num_workers,
            image_size=image_size,
            num_output_channels=num_output_channels,
            k_folds=K_FOLDS,
            lr_for_path=LR,
            wd_for_path=WD,
        )
    encoder_ckpt, probe_ckpt = resolve_teacher_paths(
        dataset_name,
        experiment_name,
        teacher_name,
        k,
        ar,
        eval_root=eval_root,
        output_root=output_root,
        lr=LR,
        wd=WD,
        train_ratio=TRAIN_RATIO,
        data_seed=DATA_SEED,
    )
    if not probe_ckpt.is_file():
        raise FileNotFoundError(f"Teacher probe checkpoint not found: {probe_ckpt}")

    if teacher_name == "ViTMAE":
        if encoder_ckpt is None or not encoder_ckpt.is_file():
            raise FileNotFoundError(f"ViTMAE encoder checkpoint not found: {encoder_ckpt}")
        model = CustomViTMAE(
            num_classes=NUM_CLASSES, device=device,
            vitmae_checkpoint_path=str(encoder_ckpt),
            classifier_checkpoint_path=str(probe_ckpt),
        )
    elif teacher_name == "MII":
        model = CustomMII(
            num_classes=NUM_CLASSES, device=device,
            classifier_checkpoint_path=str(probe_ckpt),
        )
    else:
        raise ValueError(f"Unknown teacher: {teacher_name}")

    model = model.to(device)
    model.eval()
    return model

# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _compute_pseudolabel_accuracy(teacher, dloader_l, dloader_u, device) -> float:
    """Compute teacher argmax agreement with ground truth across all training data."""
    correct = 0
    total = 0
    teacher.eval()
    with torch.no_grad():
        for dloader in ([dloader_l] if dloader_u is None else [dloader_l, dloader_u]):
            for img, lbl, _ in dloader:
                image = img.float().to(device)
                soft, _ = teacher(image)
                preds = torch.argmax(soft, dim=1).cpu()
                correct += (preds == lbl.long()).sum().item()
                total += lbl.size(0)
    return correct / total if total > 0 else 0.0


def _labeled_forward(model, criterion, img, lbl, teacher, lam_l, softlabels, device,
                     ablation=False):
    """Forward pass on a labeled batch. Returns (loss_for_backward, ce_loss_scalar).

    When ablation=True the teacher signal is withheld from labeled data,
    so only the hard-label CE drives this step (teacher still trains on
    unlabeled data via the caller).
    """
    image = img.float().to(device)
    label_t = torch.unsqueeze(lbl.float(), 1).to(device)
    label_oh = torch.zeros(label_t.size(0), NUM_CLASSES, device=device)
    label_oh.scatter_(1, label_t.long().view(-1, 1), 1)
    if softlabels:
        label_oh = abs(label_oh - 0.1)

    pred, _ = model(image)
    pred_log = torch.log(pred.squeeze(dim=1).clamp(min=CLS_EPS))
    ce_loss = criterion(pred_log, label_oh)

    if teacher is not None and lam_l > 0 and not ablation:
        with torch.no_grad():
            soft, _ = teacher(image)
        soft_ce = criterion(pred_log, soft)
        loss = ce_loss + lam_l * soft_ce
    else:
        loss = ce_loss

    return loss, ce_loss.item()


def train_epoch(model, optimizer, criterion, dloader_l, dloader_u,
                teacher, lam_l, lam_u, softlabels, device, ablation=False):
    """One training epoch. Returns average CE loss on labeled data."""
    model.train()
    ce_total = 0.0
    n = 0

    if dloader_u is None:
        # ar = 1.0: all data is labeled
        for img, lbl, _ in dloader_l:
            optimizer.zero_grad()
            loss, ce = _labeled_forward(
                model, criterion, img, lbl, teacher, lam_l, softlabels, device,
                ablation=ablation,
            )
            loss.backward()
            optimizer.step()
            ce_total += ce
            n += 1
    else:
        # ar < 1.0: interleaved labeled + unlabeled (matches main.py indep approach)
        if len(dloader_l.dataset) >= len(dloader_u.dataset):
            outer, inner, outer_labeled = dloader_l, dloader_u, True
        else:
            outer, inner, outer_labeled = dloader_u, dloader_l, False

        inner_iter = iter(inner)
        for outer_batch in outer:
            try:
                inner_batch = next(inner_iter)
            except StopIteration:
                break

            l_batch = outer_batch if outer_labeled else inner_batch
            u_batch = inner_batch if outer_labeled else outer_batch

            # Step 1: labeled update (ablation skips teacher here)
            optimizer.zero_grad()
            loss, ce = _labeled_forward(
                model, criterion, l_batch[0], l_batch[1],
                teacher, lam_l, softlabels, device,
                ablation=ablation,
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Step 2: unlabeled update
            if teacher is not None and lam_u > 0:
                img_u = u_batch[0].float().to(device)
                pred, _ = model(img_u)
                pred_log = torch.log(pred.squeeze(dim=1).clamp(min=CLS_EPS))
                with torch.no_grad():
                    soft_u, _ = teacher(img_u)
                loss_u = lam_u * criterion(pred_log, soft_u)
                loss_u.backward()
            optimizer.step()

            ce_total += ce
            n += 1

    return ce_total / max(n, 1)


def validate(model, criterion, dloader, softlabels, device) -> float:
    """Returns average CE loss on validation set."""
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for img, lbl, _ in dloader:
            image = img.float().to(device)
            label_t = torch.unsqueeze(lbl.float(), 1).to(device)
            label_oh = torch.zeros(label_t.size(0), NUM_CLASSES, device=device)
            label_oh.scatter_(1, label_t.long().view(-1, 1), 1)
            if softlabels:
                label_oh = abs(label_oh - 0.1)
            pred, _ = model(image)
            pred_log = torch.log(pred.squeeze(dim=1).clamp(min=CLS_EPS))
            total += criterion(pred_log, label_oh).item()
            n += 1
    return total / max(n, 1)

# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------

def evaluate_test(model, dloader, device) -> dict:
    """Run test set once, return top-1 accuracy, AUROC, and per-sample predictions."""
    auroc_metric = BinaryAUROC()
    model.eval()

    all_y_true: list[float] = []
    all_y_prob: list[float] = []
    all_pred_log: list[torch.Tensor] = []
    all_label_oh: list[torch.Tensor] = []

    with torch.no_grad():
        for img, lbl, _ in dloader:
            image = img.float().to(device)
            label_t = torch.unsqueeze(lbl.float(), 1).to(device)
            label_oh = torch.zeros(label_t.size(0), NUM_CLASSES, device=device)
            label_oh.scatter_(1, label_t.long().view(-1, 1), 1)

            pred, _ = model(image)
            pred_log = torch.log(pred.squeeze(dim=1).clamp(min=CLS_EPS))

            all_y_prob.extend(pred[:, 1].detach().cpu().tolist())
            all_y_true.extend(lbl.float().tolist())
            all_pred_log.append(pred_log.cpu())
            all_label_oh.append(label_oh.cpu())

    auc = float(auroc_metric(tensor(all_y_prob), tensor(all_y_true)).item())

    # Top-1 accuracy (same method as main.py for consistency)
    score_t = torch.cat(all_pred_log, dim=0)
    label_t = torch.cat(all_label_oh, dim=0)
    _, y_true_idx = torch.topk(label_t, k=1, dim=1)
    _, y_pred_idx = torch.topk(score_t, k=NUM_CLASSES, dim=1)
    top1 = float(torch.sum(y_true_idx == y_pred_idx[:, :1]).item()) / y_true_idx.size(0)

    return {
        "top1": top1,
        "auroc": auc,
        "y_true": np.array(all_y_true),
        "y_prob": np.array(all_y_prob),
    }

# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------

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
) -> dict:
    """Train one fold, freeze at best val loss, evaluate test set once."""
    ar_dir = Path(eval_root) / preset.dataset_name / experiment.name / f"ar{ar}"
    ar_dir.mkdir(parents=True, exist_ok=True)

    # -- Data (same seeds and splits as main.py) --
    ssl = (ar != 1.0)
    dataset_path = f"./datasets/{preset.dataset_name}"

    data_gen = seed_everything(DATA_SEED)

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

    print(f"  [k={k}] train_l={len(dloader_l.dataset)}"
          f" train_u={len(dloader_u.dataset) if dloader_u else 0}"
          f" valid={len(valid_dloader.dataset)} test={len(test_dloader.dataset)}")

    # -- Teacher --
    teacher = None
    pseudolabel_acc = None
    if experiment.teacher:
        teacher = load_teacher(
            experiment.teacher,
            preset.dataset_name,
            experiment.name,
            k,
            ar,
            device,
            eval_root=eval_root,
            output_root=output_root,
            dataset_path=dataset_path,
            class_sizes=list(preset.class_sizes),
            softlabels=experiment.softlabels,
            image_size=preset.image_size,
            num_output_channels=preset.num_output_channels,
            num_workers=num_workers,
        )
        pseudolabel_acc = _compute_pseudolabel_accuracy(teacher, dloader_l, dloader_u, device)
        print(f"  [k={k}] Loaded teacher: {experiment.teacher}"
              f"  pseudolabel_acc={pseudolabel_acc:.4f}")

    # -- Student model (seed before creation, same as main.py) --
    seed_everything(MODEL_SEED)

    model = CustomResNet(
        num_channels=preset.num_output_channels,
        num_classes=NUM_CLASSES,
        pretrained=False,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    criterion = ClsCriterion()

    # -- TensorBoard logger (one per fold, matching main.py) --
    log_dir = ar_dir / f"k{k}"
    logger = SummaryWriter(log_dir=str(log_dir))

    # -- Training loop with validation-based checkpoint selection --
    best_val_loss = math.inf
    best_epoch = -1

    for epoch in range(epochs):
        lam = alpha_schedule(0, epoch, LAMBDA_RAMP_EPOCHS, experiment.lam) if experiment.teacher else 0.0

        train_ce = train_epoch(
            model, optimizer, criterion,
            dloader_l, dloader_u, teacher,
            lam_l=lam, lam_u=lam,
            softlabels=experiment.softlabels,
            device=device,
            ablation=experiment.ablation,
        )
        val_loss = validate(model, criterion, valid_dloader, experiment.softlabels, device)

        logger.add_scalar("Train/CE Loss", train_ce, epoch + 1)
        logger.add_scalar("Valid/CE Losses", val_loss, epoch + 1)

        if math.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            save_checkpoint(str(ar_dir), {
                "epoch": epoch + 1,
                "best_loss": best_val_loss,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, filename=f"k{k}_best")

        if (epoch + 1) % 50 == 0 or epoch == 0:
            val_str = f"{val_loss:.4f}" if math.isfinite(val_loss) else str(val_loss)
            train_str = f"{train_ce:.4f}" if math.isfinite(train_ce) else str(train_ce)
            print(f"  [k={k}] epoch {epoch+1:>3}/{epochs}"
                  f"  train_ce={train_str}  val_ce={val_str}"
                  f"  best@{best_epoch+1}")

    # -- Load best checkpoint and evaluate test set --
    ckpt_path = ar_dir / "checkpoints" / f"k{k}_best.pth.tar"
    if not ckpt_path.is_file():
        raise RuntimeError(
            f"No best checkpoint was written to {ckpt_path}. "
            "Validation loss was never a finite improvement over the previous best "
            "(often caused by NaN tensors in the val pipeline). "
            "If this persists after updating MinMaxNorm, inspect data transforms and labels."
        )
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"  [k={k}] Loaded best checkpoint (epoch {ckpt['epoch']},"
          f" val_loss={ckpt['best_loss']:.4f})")

    results = evaluate_test(model, test_dloader, device)

    logger.add_scalar("Test/Top 1 accuracy", results["top1"], 1)
    logger.add_scalar("Test/AUROC", results["auroc"], 1)

    logger.flush()
    logger.close()

    # Save predictions in phair cross-validation format
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
    eval_root: str = "eval_output",
    output_root: str = "output",
    num_workers: int = NUM_WORKERS,
):
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment.name}  |  Dataset: {preset.dataset_name}")
    print(f"Teacher: {experiment.teacher or 'None'}  |  lambda: {experiment.lam}"
          f"  |  softlabels: {experiment.softlabels}"
          f"  |  ablation: {experiment.ablation}")
    print(f"Epochs: {epochs}  |  Device: {device}")
    print(f"{'='*60}")

    cross_ar_rows: list[dict] = []

    for ar in ar_list:
        print(f"\n--- ar={ar} ---")
        fold_results: list[dict] = []

        for k in range(K_FOLDS):
            res = run_fold(
                preset, experiment, ar, k, device, epochs, eval_root, output_root,
                num_workers=num_workers,
            )
            fold_results.append({
                "k": k, "top1": res["top1"], "auroc": res["auroc"],
                "pseudolabel_acc": res["pseudolabel_acc"],
            })

        # Per-AR summary across folds
        top1s = [r["top1"] for r in fold_results]
        aurocs = [r["auroc"] for r in fold_results]
        pl_accs = [r["pseudolabel_acc"] for r in fold_results
                   if r["pseudolabel_acc"] is not None]
        summary_mean = {"k": "mean", "top1": np.mean(top1s), "auroc": np.mean(aurocs)}
        summary_std = {
            "k": "std",
            "top1": np.std(top1s) / np.sqrt(K_FOLDS),
            "auroc": np.std(aurocs) / np.sqrt(K_FOLDS),
        }
        if pl_accs:
            summary_mean["pseudolabel_acc"] = np.mean(pl_accs)
            summary_std["pseudolabel_acc"] = np.std(pl_accs) / np.sqrt(K_FOLDS)

        ar_dir = Path(eval_root) / preset.dataset_name / experiment.name / f"ar{ar}"

        # PHAIR evaluation (writes fold_k/metrics.json used below)
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

        print(f"  ar={ar} complete:"
              f" top1={np.mean(top1s):.4f} +/- {np.std(top1s)/np.sqrt(K_FOLDS):.4f}"
              f" auroc={np.mean(aurocs):.4f} +/- {np.std(aurocs)/np.sqrt(K_FOLDS):.4f}")

    # Cross-AR summary
    exp_dir = Path(eval_root) / preset.dataset_name / experiment.name
    summary_path = exp_dir / "summary_all_ar.csv"
    pd.DataFrame(cross_ar_rows).to_csv(summary_path, index=False)
    print(f"\nSaved cross-AR summary: {summary_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_ar_list(s: str) -> list[float]:
    """Parse --ar as one float or comma-separated floats (e.g. '0.5,0.7,1.0')."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--ar must contain at least one number")
    out: list[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"invalid annotation ratio {p!r}") from e
    return out


def main_cli():
    parser = argparse.ArgumentParser(
        description="Train and evaluate student models with proper "
                    "train/validate/freeze/test protocol.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --dataset hip --experiment ResNet18_ViTMAE_l1.0 --ar 1.0 --gpu 7
  python evaluate.py --dataset hip --experiment ResNet18_MII_l1.0 --ar 0.5,0.6,0.7,0.8,0.9,1.0 --gpu 7
  python evaluate.py --dataset hip --experiment ResNet18_MII_l1.0 --gpu 7 --num-workers 2
  python evaluate.py --dataset hip --experiment ResNet18_ViTMAE_l1.0 --gpu 2,3
  python evaluate.py --dataset hip
  python evaluate.py --dataset covid --experiment ResNet18_MII_l1.0 --ar 1.0 --gpu 0
""",
    )
    parser.add_argument(
        "--dataset", required=True, choices=list(DATASET_PRESETS.keys()),
        help="Dataset preset",
    )
    parser.add_argument(
        "--experiment", default=None, choices=list(EXPERIMENTS.keys()),
        help="Experiment name (omit to run all)",
    )
    parser.add_argument(
        "--ar", type=str, default=None,
        metavar="RATIO[,RATIO...]",
        help="Annotation ratio(s): one float or comma-separated list, e.g. 1.0 or "
             "0.5,0.6,0.7,0.8,0.9,1.0 (omit for full sweep 0.1-1.0)",
    )
    parser.add_argument(
        "--eval-root", default="eval_output",
        help="Output root directory (default: eval_output)",
    )
    parser.add_argument(
        "--output-root", default="output",
        help="Training output root for teacher checkpoints (default: output)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device override (default: auto-detect)",
    )
    parser.add_argument(
        "--gpu", default=None, type=str,
        help="CUDA_VISIBLE_DEVICES value, e.g. '0', '1', '2', '0,1' (default: all visible)",
    )
    parser.add_argument(
        "--epochs", type=int, default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--num-workers", type=int, default=NUM_WORKERS,
        help=f"DataLoader workers (default: {NUM_WORKERS}). "
             "Reduce to 2 or 0 for large teacher models like MII.",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

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

    print(f"Device: {device}")
    print(f"Experiments: {[e.name for e in experiments]}")
    print(f"Annotation ratios: {ar_list}")

    for exp in experiments:
        run_experiment(
            preset, exp, ar_list, device,
            epochs=args.epochs,
            eval_root=args.eval_root,
            output_root=args.output_root,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main_cli()
