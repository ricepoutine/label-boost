"""
finetune.py - ViT-MAE unsupervised reconstruction pretraining.

Performs masked-autoencoder pretraining of facebook/vit-mae-large on a
medical-image dataset using the existing dataloader pipeline.  The
resulting encoder checkpoint is the teacher used by ``probe.py`` /
``evaluate.py`` to extract pseudolabels for student-model training.

Output layout (matches the path expected by ``probe.resolve_teacher_paths``
and ``evaluate.load_teacher``)::

    output/{dataset_name}/lr{lr}_wd{wd}/ViTMAE/tr{tr}_s{seed}_k{k}/
        checkpoints/best.pth.tar
        events.out.tfevents.*               (TensorBoard logs)

Examples::

    python finetune.py --dataset hip --epochs 150 --gpu 0
    python finetune.py --dataset covid --expanded-train-set --gpu 1
    python finetune.py --dataset hip --resume best --epochs 200
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import ViTMAEForPreTraining

from lib.dataloader import create_dataloaders
from lib.utils.avgmeter import AverageMeter
from lib.utils.model_utils import save_checkpoint
from lib.utils.plot import visualize

# ---------------------------------------------------------------------------
# Defaults (mirror main_transformer.py for ViTMAE pretraining)
# ---------------------------------------------------------------------------
DEFAULT_EPOCHS = 150
DEFAULT_LR = 1e-4
DEFAULT_WD = 1e-5
DEFAULT_BATCH_SIZE = 64
NUM_WORKERS = 8
TRAIN_RATIO = 0.7
VALID_RATIO = 0.1
DATA_SEED = 0
K_FOLDS = 5
DEFAULT_MANUAL_CHECKPOINT = 50  # only save best after this epoch
RECON_VIS_FREQ = 1               # how often (in epochs) to log MAE reconstructions

# ---------------------------------------------------------------------------
# Dataset presets (mirror evaluate.DATASET_PRESETS for the binary datasets we
# pretrain on).  Image size is fixed at 224x224 because facebook/vit-mae-large
# expects 224 patches.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetPreset:
    dataset_name: str
    class_sizes: tuple[int, int] = (2000, 2000)
    image_size: tuple[int, int] = (224, 224)
    num_output_channels: int = 3


DATASET_PRESETS: dict[str, DatasetPreset] = {
    "hip": DatasetPreset("hip734_cropped"),
    "covid": DatasetPreset("Covid"),
}


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
# Output paths
# ---------------------------------------------------------------------------

def run_dir(output_root: str, dataset_name: str, lr: float, wd: float,
            tr: float, seed: int, k: int) -> Path:
    return (
        Path(output_root)
        / dataset_name
        / f"lr{lr}_wd{wd}"
        / "ViTMAE"
        / f"tr{tr}_s{seed}_k{k}"
    )


# ---------------------------------------------------------------------------
# Train / test loops (extracted from main_transformer.train / .test)
# ---------------------------------------------------------------------------

def train_one_epoch(model, dloader, optimizer, logger, epoch, device,
                    batch_size, batched=True, print_freq=10,
                    recon_vis_freq=RECON_VIS_FREQ):
    model.train()
    optimizer.zero_grad()
    losses = AverageMeter()
    image = None
    outputs = None
    for i, (img, _, _) in enumerate(dloader):
        image = img.float().to(device)
        outputs = model(image)
        loss = outputs.loss
        losses.update(float(loss.item()), batch_size)
        if i % print_freq == 0:
            print(f"  [train] epoch={epoch+1} batch={i} loss={loss.item():.4f}")
            if torch.isnan(loss):
                return False, losses.avg
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if image is not None and outputs is not None and (epoch % recon_vis_freq == 0):
        sample = image[0].unsqueeze(0) if batched else image
        with torch.no_grad():
            _, reconstruction, both = visualize(
                sample.detach().cpu(), outputs, model,
                batched=batched, show_images=False,
            )
        logger.add_image("Train/Original", sample, epoch + 1, dataformats="NCHW")
        logger.add_image("Train/Reconstruction", reconstruction, epoch + 1,
                         dataformats="HWC")
        logger.add_image("Train/Recon + Visible", both, epoch + 1,
                         dataformats="HWC")
        logger.flush()

    logger.add_scalar("Train/Recon loss", losses.avg, epoch + 1)
    return True, losses.avg


def evaluate_recon(model, dloader, logger, epoch, device, batch_size,
                   batched=True, print_freq=10, recon_vis_freq=RECON_VIS_FREQ,
                   tag="Test"):
    model.eval()
    losses = AverageMeter()
    image = None
    outputs = None
    with torch.no_grad():
        for i, (img, _, _) in enumerate(dloader):
            image = img.float().to(device)
            outputs = model(image)
            loss = outputs.loss
            losses.update(float(loss.item()), batch_size)
            if i % print_freq == 0:
                print(f"  [{tag.lower()}] epoch={epoch+1} batch={i} loss={loss.item():.4f}")

    if image is not None and outputs is not None and (epoch % recon_vis_freq == 0):
        sample = image[0].unsqueeze(0) if batched else image
        with torch.no_grad():
            _, reconstruction, both = visualize(
                sample.detach().cpu(), outputs, model,
                batched=batched, show_images=False,
            )
        if epoch == 0:
            logger.add_image(f"{tag}/Original", sample, epoch + 1,
                             dataformats="NCHW")
        logger.add_image(f"{tag}/Reconstruction", reconstruction, epoch + 1,
                         dataformats="HWC")
        logger.add_image(f"{tag}/Recon + Visible", both, epoch + 1,
                         dataformats="HWC")
        logger.flush()

    logger.add_scalar(f"{tag}/Recon loss", losses.avg, epoch + 1)
    return losses.avg


# ---------------------------------------------------------------------------
# One fold of pretraining
# ---------------------------------------------------------------------------

def run_fold(
    preset: DatasetPreset,
    *,
    k: int,
    epochs: int,
    lr: float,
    wd: float,
    batch_size: int,
    expanded_train_set: bool,
    output_root: str,
    device: torch.device,
    num_workers: int,
    manual_checkpoint: int,
    resume: Optional[str],
    overwrite_existing_log: bool,
):
    out_dir = run_dir(output_root, preset.dataset_name, lr, wd,
                      TRAIN_RATIO, DATA_SEED, k)
    chk_dir = out_dir / "checkpoints"

    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-large")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # -- resume / overwrite handling (mirrors main_transformer.main) --
    start_epoch = 0
    if out_dir.exists():
        print(f"  [k={k}] output dir exists: {out_dir}")
        if chk_dir.exists() and resume in ("latest", "best", "100", "150", "manual"):
            ckpt_path = chk_dir / f"{resume}.pth.tar"
            if ckpt_path.is_file():
                print(f"  [k={k}] resuming from {ckpt_path}")
                weights = torch.load(str(ckpt_path), map_location="cpu")
                model.load_state_dict(weights["state_dict"])
                optimizer.load_state_dict(weights["optimizer"])
                start_epoch = int(weights["epoch"])
                weights = None
            else:
                print(f"  [k={k}] requested resume={resume} not found at {ckpt_path}")
                return
        elif overwrite_existing_log and not chk_dir.exists():
            print(f"  [k={k}] removing existing log dir (no checkpoint inside)")
            shutil.rmtree(out_dir, ignore_errors=True)
        elif not chk_dir.exists():
            out_dir = Path(str(out_dir) + "_" + str(uuid.uuid4()))
            chk_dir = out_dir / "checkpoints"
            print(f"  [k={k}] writing fresh logs to {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # -- Data --
    dataset_path = f"./datasets/{preset.dataset_name}"
    if expanded_train_set:
        # use original full training set (label stripped) for MAE pretraining
        _, valid_dloader, test_dloader, train_dloader = create_dataloaders(
            dataset_name=preset.dataset_name,
            dataset_path=dataset_path,
            class_sizes=list(preset.class_sizes),
            seed=DATA_SEED,
            train_ratio=TRAIN_RATIO,
            valid_ratio=VALID_RATIO,
            annot_ratio=1.0,
            batch_size=batch_size,
            num_workers=num_workers,
            num_output_channels=preset.num_output_channels,
            image_size=preset.image_size,
            k_folds=K_FOLDS,
            k_fold_index=k,
            keep_remainder=True,
        )
        print(f"  [k={k}] expanded train set: {len(train_dloader.dataset)}")
    else:
        train_dloader, valid_dloader, test_dloader = create_dataloaders(
            dataset_name=preset.dataset_name,
            dataset_path=dataset_path,
            class_sizes=list(preset.class_sizes),
            seed=DATA_SEED,
            train_ratio=TRAIN_RATIO,
            valid_ratio=VALID_RATIO,
            annot_ratio=1.0,
            batch_size=batch_size,
            num_workers=num_workers,
            num_output_channels=preset.num_output_channels,
            image_size=preset.image_size,
            k_folds=K_FOLDS,
            k_fold_index=k,
            keep_remainder=False,
        )

    print(f"  [k={k}] train={len(train_dloader.dataset)}"
          f" valid={len(valid_dloader.dataset)}"
          f" test={len(test_dloader.dataset)}")

    logger = SummaryWriter(log_dir=str(out_dir))
    best_loss = math.inf
    batched = batch_size > 1

    try:
        for epoch in range(start_epoch, epochs):
            if device.type == "cuda":
                torch.cuda.empty_cache()

            ok, _train_loss = train_one_epoch(
                model, train_dloader, optimizer, logger, epoch, device,
                batch_size, batched=batched,
            )
            if not ok:
                print(f"  [k={k}] aborting fold (NaN train loss)")
                return

            val_loss = evaluate_recon(
                model, valid_dloader, logger, epoch, device, batch_size,
                batched=batched, tag="Valid",
            )
            evaluate_recon(
                model, test_dloader, logger, epoch, device, batch_size,
                batched=batched, tag="Test",
            )

            if val_loss < best_loss and (epoch + 1) > manual_checkpoint:
                best_loss = val_loss
                save_checkpoint(
                    str(out_dir),
                    {
                        "epoch": epoch + 1,
                        "best_loss": best_loss,
                        "training params": {
                            "batch_size": batch_size,
                            "lr": lr,
                            "wd": wd,
                            "epochs": epochs,
                        },
                        "model params": {"base model": "ViTMAE"},
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    filename="best",
                )
                print(f"  [k={k}] epoch {epoch+1}: new best valid recon = {best_loss:.4f}")
    finally:
        logger.flush()
        logger.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_k_list(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("--k must contain at least one fold index")
    out: list[int] = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"invalid fold index {p!r}") from e
    return out


def main_cli():
    parser = argparse.ArgumentParser(
        description="ViT-MAE unsupervised reconstruction pretraining.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python finetune.py --dataset hip --epochs 150 --gpu 0
  python finetune.py --dataset covid --expanded-train-set --gpu 1
  python finetune.py --dataset hip --k 0,1,2 --epochs 150
  python finetune.py --dataset hip --resume best --epochs 200
""",
    )
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASET_PRESETS.keys()),
                        help="Dataset preset")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help=f"Pretraining epochs (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help=f"Learning rate (default: {DEFAULT_LR})")
    parser.add_argument("--wd", type=float, default=DEFAULT_WD,
                        help=f"Weight decay (default: {DEFAULT_WD})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help=f"DataLoader workers (default: {NUM_WORKERS})")
    parser.add_argument("--k", type=parse_k_list, default=None,
                        metavar="IDX[,IDX...]",
                        help="Fold index/indices to run (default: 0..K_FOLDS-1)")
    parser.add_argument("--expanded-train-set", action="store_true",
                        help="Include held-out remainder samples in MAE training "
                             "(self-supervised, no labels needed)")
    parser.add_argument("--manual-checkpoint", type=int,
                        default=DEFAULT_MANUAL_CHECKPOINT,
                        help="Only save 'best' once epoch+1 exceeds this value")
    parser.add_argument("--resume", choices=["latest", "best", "100", "150", "manual"],
                        default=None,
                        help="Resume from a checkpoint of this name (if present)")
    parser.add_argument("--overwrite", action="store_true",
                        help="If output dir exists with no checkpoint, remove it")
    parser.add_argument("--output-root", default="output",
                        help="Output root directory (default: output)")
    parser.add_argument("--device", default=None,
                        help="Device override (default: auto-detect)")
    parser.add_argument("--gpu", default=None, type=str,
                        help="CUDA_VISIBLE_DEVICES value (e.g. '0', '0,1')")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    preset = DATASET_PRESETS[args.dataset]
    device = torch.device(args.device) if args.device else default_device()
    folds = args.k if args.k is not None else list(range(K_FOLDS))

    print("=" * 60)
    print(f"ViTMAE pretraining | dataset={preset.dataset_name}")
    print(f"epochs={args.epochs}  lr={args.lr}  wd={args.wd}"
          f"  batch_size={args.batch_size}")
    print(f"folds={folds}  expanded_train_set={args.expanded_train_set}"
          f"  device={device}")
    print("=" * 60)

    for k in folds:
        print(f"\n--- fold k={k} ---")
        run_fold(
            preset,
            k=k,
            epochs=args.epochs,
            lr=args.lr,
            wd=args.wd,
            batch_size=args.batch_size,
            expanded_train_set=args.expanded_train_set,
            output_root=args.output_root,
            device=device,
            num_workers=args.num_workers,
            manual_checkpoint=args.manual_checkpoint,
            resume=args.resume,
            overwrite_existing_log=args.overwrite,
        )


if __name__ == "__main__":
    main_cli()
