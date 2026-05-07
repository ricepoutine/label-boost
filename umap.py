"""
umap.py - UMAP projections of medical-image features (PNG output).

Renders the three projection figures used in the paper:

    figures/umap/mii_untrained_umap.png      (MedImageInsights, untrained)
    figures/umap/vitmae_base_umap.png        (facebook/vit-mae-large, untrained)
    figures/umap/vitmae_finetuned_umap.png   (ViT-MAE after finetune.py)

Model checkpoints are loaded with the same path conventions as
``finetune.py`` / ``probe.py`` (see ``run_dir`` below).

Examples::

    python umap.py --all                                      # all three thesis figures (hip)
    python umap.py --dataset hip --model vitmae --variant finetuned --k 0
    python umap.py --dataset hip --model vitmae --variant base
    python umap.py --dataset hip --model mii   --variant untrained
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# This file is intentionally named ``umap.py`` per the project convention.
# When the project root sits on ``sys.path`` (it does whenever this script is
# invoked via ``python umap.py`` or imported from ``generate_results.py``),
# a bare ``import umap`` would re-resolve to *this* module instead of the
# installed ``umap-learn`` library.  Strip our directory from ``sys.path``
# for the duration of the upstream import, then restore it.
_HERE = os.path.dirname(os.path.abspath(__file__))


def _is_local_path(p: str) -> bool:
    return os.path.abspath(p or os.getcwd()) == _HERE


_DROPPED_PATHS = [(i, p) for i, p in enumerate(sys.path) if _is_local_path(p)]
for _i, _ in reversed(_DROPPED_PATHS):
    sys.path.pop(_i)
sys.modules.pop("umap", None)
import umap as umap_lib  # noqa: E402  # third-party umap-learn
for _i, _p in _DROPPED_PATHS:
    sys.path.insert(_i, _p)
del _DROPPED_PATHS

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import ViTMAEForPreTraining  # noqa: E402

from lib.dataloader import create_dataloaders  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults (must match the path layout used by finetune.py / evaluate.py)
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.7
VALID_RATIO = 0.1
DATA_SEED = 0
K_FOLDS = 5
NUM_WORKERS = 8
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-4    # finetune.py default for ViTMAE pretraining
DEFAULT_WD = 1e-5    # finetune.py default for ViTMAE pretraining

# ---------------------------------------------------------------------------
# Dataset presets (match evaluate.DATASET_PRESETS / finetune.DATASET_PRESETS)
# ---------------------------------------------------------------------------

DATASET_PRESETS: dict[str, dict] = {
    "hip": {"dataset_name": "hip734_cropped", "class_sizes": (2000, 2000)},
    "covid": {"dataset_name": "Covid",        "class_sizes": (2000, 2000)},
}

LABEL_MAP = {0: "dysplastic", 1: "normal"}
COLOR_MAP = {"dysplastic": "tab:blue", "normal": "tab:red"}

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
# Path helpers
# ---------------------------------------------------------------------------

def vitmae_run_dir(output_root: str, dataset_name: str, lr: float, wd: float,
                   tr: float, seed: int, k: int) -> Path:
    return (
        Path(output_root)
        / dataset_name
        / f"lr{lr}_wd{wd}"
        / "ViTMAE"
        / f"tr{tr}_s{seed}_k{k}"
    )


# ---------------------------------------------------------------------------
# Feature extractors (mirror projections.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def vitmae_features(model, dloader, device) -> tuple[np.ndarray, np.ndarray]:
    feats: list[torch.Tensor] = []
    labels: list[float] = []
    for image, label, _ in dloader:
        image = image.float().to(device)
        outputs = model(image, output_hidden_states=True)
        # Sum across patch-token dimension (per-image embedding).
        f = torch.sum(outputs.hidden_states[-1].cpu(), dim=1)  # [B, 1024]
        feats.append(f)
        labels.extend(label.tolist())
    X = torch.cat(feats, dim=0).numpy()
    y = np.array(labels)
    return X, y


@torch.no_grad()
def mii_features(model, dloader, device, norm=False) -> tuple[np.ndarray, np.ndarray]:
    feats: list[torch.Tensor] = []
    labels: list[float] = []
    for image, label, _ in dloader:
        image = image.float().to(device)
        out = model.get_image_embeddings(image, norm=norm)
        f = torch.as_tensor(out).reshape(image.shape[0], -1).cpu()
        feats.append(f)
        labels.extend(label.tolist())
    X = torch.cat(feats, dim=0).numpy()
    y = np.array(labels)
    return X, y


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def render_umap(X: np.ndarray, y: np.ndarray, *, title: str, out_path: Path,
                seed: int = 0, n_neighbors: int = 15, min_dist: float = 0.1):
    reducer = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                            random_state=seed)
    embedding = reducer.fit_transform(X)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_idx, cls_name in LABEL_MAP.items():
        mask = (y == cls_idx)
        if not np.any(mask):
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            s=14,
            c=COLOR_MAP.get(cls_name, "grey"),
            label=cls_name,
            alpha=0.85,
            edgecolors="none",
        )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("First UMAP")
    ax.set_ylabel("Second UMAP")
    ax.legend(title="class", loc="best")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved UMAP: {out_path}")


# ---------------------------------------------------------------------------
# Public entry points (also used by generate_results.py)
# ---------------------------------------------------------------------------

def _build_dataloader(preset_name: str, *, ar: float, k: int,
                      batch_size: int, num_workers: int, image_size=(224, 224)):
    p = DATASET_PRESETS[preset_name]
    dataset_name = p["dataset_name"]
    dataset_path = f"./datasets/{dataset_name}"
    ssl = ar != 1.0
    loaders = create_dataloaders(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        class_sizes=list(p["class_sizes"]),
        seed=DATA_SEED,
        train_ratio=TRAIN_RATIO,
        valid_ratio=VALID_RATIO,
        annot_ratio=ar,
        batch_size=batch_size,
        num_workers=num_workers,
        num_output_channels=3,
        image_size=image_size,
        k_folds=K_FOLDS,
        k_fold_index=k,
    )
    if ssl:
        _, train_l, _, _ = loaders
    else:
        train_l, _, _ = loaders
    return train_l


def render_vitmae(*, dataset: str, variant: str, k: int, ar: float,
                  out_dir: Path, device: torch.device,
                  lr: float = DEFAULT_LR, wd: float = DEFAULT_WD,
                  output_root: str = "output",
                  batch_size: int = DEFAULT_BATCH_SIZE,
                  num_workers: int = NUM_WORKERS) -> Path:
    """Render figures/umap/vitmae_{base,finetuned}_umap.png."""
    if variant not in ("base", "finetuned"):
        raise ValueError(f"variant must be 'base' or 'finetuned', got {variant}")

    model = ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-large")
    model = model.to(device)

    if variant == "finetuned":
        ckpt = vitmae_run_dir(
            output_root, DATASET_PRESETS[dataset]["dataset_name"],
            lr, wd, TRAIN_RATIO, DATA_SEED, k,
        ) / "checkpoints" / "best.pth.tar"
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"ViT-MAE finetuned checkpoint not found: {ckpt}\n"
                "Run finetune.py first or pass --output-root / --lr / --wd "
                "to point at an existing pretraining run."
            )
        weights = torch.load(str(ckpt), map_location=device)
        model.load_state_dict(weights["state_dict"])

    model.eval()
    dloader = _build_dataloader(dataset, ar=ar, k=k, batch_size=batch_size,
                                num_workers=num_workers)
    X, y = vitmae_features(model, dloader, device)

    title = ("UMAP of ViT-MAE features (pretrained)"
             if variant == "base"
             else "UMAP of ViT-MAE features (finetuned)")
    out_path = out_dir / f"vitmae_{variant}_umap.png"
    render_umap(X, y, title=title, out_path=out_path)
    return out_path


def render_mii(*, dataset: str, k: int, ar: float, out_dir: Path,
               device: torch.device,
               batch_size: int = DEFAULT_BATCH_SIZE,
               num_workers: int = NUM_WORKERS,
               norm: bool = False) -> Path:
    """Render figures/umap/mii_untrained_umap.png."""
    # Imported lazily because MedImageInsights pulls in heavy/optional deps.
    from models.mii import CustomMII
    mii = CustomMII(num_classes=2, device=device)
    backbone = mii.features  # MedImageInsight wrapper exposing get_image_embeddings

    dloader = _build_dataloader(dataset, ar=ar, k=k, batch_size=batch_size,
                                num_workers=num_workers)
    X, y = mii_features(backbone, dloader, device, norm=norm)

    title = "UMAP of MedImageInsights features (untrained)"
    out_path = out_dir / "mii_untrained_umap.png"
    render_umap(X, y, title=title, out_path=out_path)
    return out_path


def render_all(*, dataset: str, k: int, ar: float, out_dir: Path,
               device: torch.device,
               lr: float = DEFAULT_LR, wd: float = DEFAULT_WD,
               output_root: str = "output",
               batch_size: int = DEFAULT_BATCH_SIZE,
               num_workers: int = NUM_WORKERS) -> list[Path]:
    """Render the three figures used in the thesis figures/umap/ folder."""
    paths: list[Path] = []
    paths.append(render_mii(
        dataset=dataset, k=k, ar=ar, out_dir=out_dir, device=device,
        batch_size=batch_size, num_workers=num_workers,
    ))
    paths.append(render_vitmae(
        dataset=dataset, variant="base", k=k, ar=ar, out_dir=out_dir,
        device=device, lr=lr, wd=wd, output_root=output_root,
        batch_size=batch_size, num_workers=num_workers,
    ))
    paths.append(render_vitmae(
        dataset=dataset, variant="finetuned", k=k, ar=ar, out_dir=out_dir,
        device=device, lr=lr, wd=wd, output_root=output_root,
        batch_size=batch_size, num_workers=num_workers,
    ))
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main_cli():
    parser = argparse.ArgumentParser(
        description="Render UMAP feature projections (PNG).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python umap.py --all
  python umap.py --dataset hip --model vitmae --variant finetuned --k 0
  python umap.py --dataset hip --model vitmae --variant base
  python umap.py --dataset hip --model mii   --variant untrained
""",
    )
    parser.add_argument("--dataset", default="hip",
                        choices=list(DATASET_PRESETS.keys()))
    parser.add_argument("--model", choices=["vitmae", "mii"], default=None)
    parser.add_argument("--variant", choices=["base", "finetuned", "untrained"],
                        default=None)
    parser.add_argument("--ar", type=float, default=1.0,
                        help="Annotation ratio of the labeled set used for the projection")
    parser.add_argument("--k", type=int, default=0, help="Fold index")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help="LR used during ViT-MAE pretraining (path lookup)")
    parser.add_argument("--wd", type=float, default=DEFAULT_WD,
                        help="WD used during ViT-MAE pretraining (path lookup)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--output-root", default="output",
                        help="Pretraining checkpoint root (default: output)")
    parser.add_argument("--out-dir", default="figures/umap",
                        help="Output directory for PNGs (default: figures/umap)")
    parser.add_argument("--all", action="store_true",
                        help="Produce mii_untrained, vitmae_base, vitmae_finetuned")
    parser.add_argument("--gpu", default=None, type=str,
                        help="CUDA_VISIBLE_DEVICES value (e.g. '0', '0,1')")
    parser.add_argument("--device", default=None,
                        help="Device override (default: auto-detect)")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device(args.device) if args.device else default_device()
    out_dir = Path(args.out_dir)

    if args.all:
        render_all(
            dataset=args.dataset, k=args.k, ar=args.ar, out_dir=out_dir,
            device=device, lr=args.lr, wd=args.wd,
            output_root=args.output_root, batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        return

    if args.model is None or args.variant is None:
        parser.error("Provide --all, or both --model and --variant")

    if args.model == "vitmae":
        if args.variant not in ("base", "finetuned"):
            parser.error("ViT-MAE supports --variant base|finetuned")
        render_vitmae(
            dataset=args.dataset, variant=args.variant, k=args.k, ar=args.ar,
            out_dir=out_dir, device=device, lr=args.lr, wd=args.wd,
            output_root=args.output_root, batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        if args.variant != "untrained":
            parser.error("MII only supports --variant untrained")
        render_mii(
            dataset=args.dataset, k=args.k, ar=args.ar, out_dir=out_dir,
            device=device, batch_size=args.batch_size,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main_cli()
