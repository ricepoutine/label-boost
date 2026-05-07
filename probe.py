"""
Linear probe training for ViTMAE / MII teachers (extracted from main_transformer patterns).

Encoder (ViTMAE only), still under pretraining output:

  {output_root}/{dataset}/lr{lr}_wd{wd}/{Teacher}/tr{tr}_s{seed}_k{k}/checkpoints/best.pth.tar

Linear probe weights live next to student folds in eval_output (same folder as k{k}_best):

  {eval_root}/{dataset}/{experiment}/ar{ar}/checkpoints/linear_k{k}_best.pth.tar

evaluate.load_teacher calls ensure_teacher_linear_probe before loading; if the linear
checkpoint is missing, probe training runs then the experiment continues.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter

from lib.criterion import ClsCriterion
from lib.dataloader import create_dataloaders
from lib.utils.avgmeter import AverageMeter
from lib.utils.model_utils import save_checkpoint
from models.mii import CustomMII
from models.vitmae import CustomViTMAE

NUM_CLASSES = 2

# Match main_transformer defaults for linear probe training
PROBE_LR = 1e-3
PROBE_WD = 1e-4
PROBE_EPOCHS = 300
PROBE_BATCH_SIZE = 64
PROBE_MANUAL_CHECKPOINT = 0  # save when val improves for epoch+1 > this (0 = from epoch 0)


def teacher_run_dir(
    output_root: str,
    dataset_name: str,
    teacher_name: str,
    k: int,
    lr: float,
    wd: float,
    train_ratio: float,
    data_seed: int,
) -> Path:
    return (
        Path(output_root)
        / dataset_name
        / f"lr{lr}_wd{wd}"
        / teacher_name
        / f"tr{train_ratio}_s{data_seed}_k{k}"
    )


def experiment_ar_dir(
    eval_root: str, dataset_name: str, experiment_name: str, ar: float,
) -> Path:
    """Same directory layout as evaluate.run_fold ar_dir (parent of per-fold logs)."""
    return Path(eval_root) / dataset_name / experiment_name / f"ar{ar}"


def probe_checkpoint_name(k: int) -> str:
    """Base filename (no extension) for save_checkpoint; pairs with k{k}_best student ckpts."""
    return f"linear_k{k}_best"


def probe_checkpoint_path(ar_dir: Path, k: int) -> Path:
    return ar_dir / "checkpoints" / f"{probe_checkpoint_name(k)}.pth.tar"


def vitmae_encoder_path(encoder_run_dir: Path) -> Path:
    return encoder_run_dir / "checkpoints" / "best.pth.tar"


def resolve_teacher_paths(
    dataset_name: str,
    experiment_name: str,
    teacher_name: str,
    k: int,
    ar: float,
    eval_root: str = "eval_output",
    output_root: str = "output",
    lr: float = 1e-3,
    wd: float = 1e-4,
    train_ratio: float = 0.7,
    data_seed: int = 0,
) -> tuple[Optional[Path], Path]:
    """Return (encoder_ckpt or None for MII, linear_probe_ckpt under eval_output)."""
    encoder_base = teacher_run_dir(
        output_root, dataset_name, teacher_name, k, lr, wd, train_ratio, data_seed
    )
    enc = vitmae_encoder_path(encoder_base) if teacher_name == "ViTMAE" else None
    ar_dir = experiment_ar_dir(eval_root, dataset_name, experiment_name, ar)
    probe = probe_checkpoint_path(ar_dir, k)
    return enc, probe


def _freeze_mii_backbone(model: CustomMII) -> None:
    feats = getattr(model, "features", None)
    if feats is not None and hasattr(feats, "parameters"):
        for p in feats.parameters():
            p.requires_grad = False


def _train_one_epoch_classifier(
    model,
    teacher_name: str,
    dloader,
    criterion,
    optimizer,
    logger,
    epoch: int,
    softlabels: bool,
    device: torch.device,
    batch_size: int,
) -> bool:
    if teacher_name == "ViTMAE":
        model.features.eval()
    elif teacher_name == "MII":
        model.classifier.train()
    optimizer.zero_grad()
    ce_losses = AverageMeter()
    for i, (image, label, _) in enumerate(dloader):
        image = image.float().to(device)
        label = torch.unsqueeze(label.float(), 1).to(device)
        label_onehot = torch.zeros(label.size(0), NUM_CLASSES, device=device).scatter_(
            1, label.to(torch.int64).view(-1, 1), 1
        )
        if softlabels:
            label_onehot = abs(label_onehot - 0.1)

        pred, _ = model(image)
        pred_log = torch.log(pred.squeeze(dim=1))
        loss = criterion(pred_log, label_onehot)
        ce_losses.update(float(loss.item()), batch_size)
        if i % 10 == 0:
            print(f"    [probe] train batch {i} loss={loss.item():.4f}")
        if torch.isnan(loss):
            return False
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    logger.add_scalar("ProbeTrain/CE", ce_losses.avg, epoch + 1)
    return True


def _eval_classifier(
    model,
    teacher_name: str,
    dloader,
    criterion,
    logger,
    epoch: int,
    device: torch.device,
    batch_size: int,
    tag: str = "Valid",
) -> float:
    if teacher_name == "ViTMAE":
        model.features.eval()
    elif teacher_name == "MII":
        model.classifier.eval()
    ce_losses = AverageMeter()
    with torch.no_grad():
        for i, (image, label, _) in enumerate(dloader):
            image = image.float().to(device)
            label = torch.unsqueeze(label.float(), 1).to(device)
            label_onehot = torch.zeros(label.size(0), NUM_CLASSES, device=device).scatter_(
                1, label.to(torch.int64).view(-1, 1), 1
            )
            pred, _ = model(image)
            pred_log = torch.log(pred.squeeze(dim=1))
            loss = criterion(pred_log, label_onehot)
            ce_losses.update(float(loss.item()), batch_size)
            if i % 10 == 0:
                print(f"    [probe] {tag} batch {i} loss={loss.item():.4f}")
    logger.add_scalar(f"Probe{tag}/CE", ce_losses.avg, epoch + 1)
    return ce_losses.avg


def train_linear_probe(
    teacher_name: str,
    dataset_name: str,
    dataset_path: str,
    class_sizes: list[int],
    k: int,
    ar: float,
    softlabels: bool,
    device: torch.device,
    encoder_run_dir: Path,
    probe_ar_dir: Path,
    train_ratio: float,
    valid_ratio: float,
    data_seed: int,
    num_workers: int,
    image_size: tuple[int, int],
    num_output_channels: int,
    k_folds: int,
    epochs: int = PROBE_EPOCHS,
    probe_lr: float = PROBE_LR,
    probe_wd: float = PROBE_WD,
    batch_size: int = PROBE_BATCH_SIZE,
) -> Path:
    """Train linear head; saves under probe_ar_dir/checkpoints/ (eval_output). Returns probe path."""
    if teacher_name not in ("ViTMAE", "MII"):
        raise ValueError(f"Unknown teacher for probe: {teacher_name}")

    probe_ar_dir.mkdir(parents=True, exist_ok=True)
    chk_dir = probe_ar_dir / "checkpoints"
    chk_dir.mkdir(parents=True, exist_ok=True)

    if teacher_name == "ViTMAE":
        enc_path = vitmae_encoder_path(encoder_run_dir)
        if not enc_path.is_file():
            raise FileNotFoundError(
                f"ViTMAE encoder checkpoint required for probe training:\n  {enc_path}\n"
                "Train or copy the pretrained encoder to this path first."
            )
        model = CustomViTMAE(
            num_classes=NUM_CLASSES,
            vitmae_checkpoint_path=str(enc_path),
            linear=True,
            classifier_checkpoint_path=None,
            device=device,
        )
    else:
        model = CustomMII(num_classes=NUM_CLASSES, device=device, classifier_checkpoint_path=None)
        _freeze_mii_backbone(model)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=probe_lr, weight_decay=probe_wd)
    criterion = ClsCriterion()

    ssl = ar != 1.0 and ar != 0.0
    loaders = create_dataloaders(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        class_sizes=class_sizes,
        seed=data_seed,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        annot_ratio=ar,
        batch_size=batch_size,
        num_workers=num_workers,
        num_output_channels=num_output_channels,
        image_size=image_size,
        k_folds=k_folds,
        k_fold_index=k,
    )
    if ssl:
        _, train_dloader, valid_dloader, test_dloader = loaders
    else:
        train_dloader, valid_dloader, test_dloader = loaders

    out_name = probe_checkpoint_name(k)
    log_root = probe_ar_dir / "tensorboard_linear_probe" / out_name
    log_root.mkdir(parents=True, exist_ok=True)
    logger = SummaryWriter(log_dir=str(log_root))

    best_loss = math.inf

    print(
        f"  [probe] Training linear probe ({teacher_name}) k={k} ar={ar} "
        f"epochs={epochs} -> {chk_dir / (out_name + '.pth.tar')}"
    )

    try:
        for epoch in range(epochs):
            ok = _train_one_epoch_classifier(
                model, teacher_name, train_dloader, criterion, optimizer,
                logger, epoch, softlabels, device, batch_size,
            )
            if not ok:
                raise RuntimeError("Probe training hit NaN loss; aborting.")

            val_loss = _eval_classifier(
                model, teacher_name, valid_dloader, criterion, logger, epoch, device, batch_size, "Valid",
            )
            _eval_classifier(
                model, teacher_name, test_dloader, criterion, logger, epoch, device, batch_size, "Test",
            )

            if val_loss < best_loss and epoch + 1 > PROBE_MANUAL_CHECKPOINT:
                state_dict = model.classifier.state_dict()
                save_checkpoint(
                    str(probe_ar_dir),
                    {
                        "epoch": epoch + 1,
                        "best_loss": best_loss,
                        "training params": {
                            "batch_size": batch_size,
                            "lr": probe_lr,
                            "wd": probe_wd,
                            "epochs": epochs,
                            "ar": ar,
                            "k": k,
                            "teacher": teacher_name,
                        },
                        "state_dict": state_dict,
                        "optimizer": optimizer.state_dict(),
                    },
                    filename=out_name,
                )
                best_loss = val_loss
                print(f"  [probe] epoch {epoch + 1}: new best val CE={val_loss:.4f}")
    finally:
        logger.close()

    out_path = probe_checkpoint_path(probe_ar_dir, k)
    if not out_path.is_file():
        raise RuntimeError(f"Probe training finished but checkpoint missing: {out_path}")
    return out_path


def ensure_teacher_linear_probe(
    teacher_name: str,
    dataset_name: str,
    experiment_name: str,
    dataset_path: str,
    class_sizes: list[int],
    k: int,
    ar: float,
    softlabels: bool,
    device: torch.device,
    eval_root: str = "eval_output",
    output_root: str = "output",
    train_ratio: float = 0.7,
    valid_ratio: float = 0.1,
    data_seed: int = 0,
    num_workers: int = 8,
    image_size: tuple[int, int] = (224, 224),
    num_output_channels: int = 3,
    k_folds: int = 5,
    probe_lr: float = PROBE_LR,
    probe_wd: float = PROBE_WD,
    lr_for_path: float = 1e-3,
    wd_for_path: float = 1e-4,
) -> Path:
    """
    If the probe checkpoint exists under eval_output for this experiment / ar / fold, return it.
    Otherwise train the linear probe and return the path.
    """
    probe_ar_dir = experiment_ar_dir(eval_root, dataset_name, experiment_name, ar)
    target = probe_checkpoint_path(probe_ar_dir, k)
    if target.is_file():
        print(f"  [probe] Using existing linear probe: {target}")
        return target

    encoder_run_dir = teacher_run_dir(
        output_root, dataset_name, teacher_name, k, lr_for_path, wd_for_path, train_ratio, data_seed
    )
    train_linear_probe(
        teacher_name=teacher_name,
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        class_sizes=class_sizes,
        k=k,
        ar=ar,
        softlabels=softlabels,
        device=device,
        encoder_run_dir=encoder_run_dir,
        probe_ar_dir=probe_ar_dir,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
        data_seed=data_seed,
        num_workers=num_workers,
        image_size=image_size,
        num_output_channels=num_output_channels,
        k_folds=k_folds,
        probe_lr=probe_lr,
        probe_wd=probe_wd,
    )
    return target
