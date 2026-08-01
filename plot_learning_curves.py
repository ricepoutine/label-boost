"""
plot_learning_curves.py - Val CE learning curves from existing TensorBoard logs.

Compares sample-type (medivision eval_output) vs label-type (label-boost
eval_output) using the already-logged ``Valid/CE Losses`` scalar.

Writes under::

    figures/{hip,covid}/learning_curves/
        val_ce_{experiment}_ar{ar}.png
        val_ce_grid_ar{ar}.png   (all_l1 experiments side-by-side)

Examples::

    python plot_learning_curves.py --dataset hip
    python plot_learning_curves.py --dataset hip --ar 0.1,0.5,1.0
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)

K_FOLDS = 5
VAL_TAG = "Valid/CE Losses"
TRAIN_TAG = "Train/CE Loss"

DISPLAY_NAMES: dict[str, str] = {
    "ResNet18": "Radiologist",
    "ResNet18_ViTMAE_l1.0": "Label-Boost (ViTMAE, λ=1.0)",
    "ResNet18_MII_l1.0": "Label-Boost (MII, λ=1.0)",
}

DEFAULT_EXPERIMENTS = [
    "ResNet18",
    "ResNet18_ViTMAE_l1.0",
    "ResNet18_MII_l1.0",
]

FIG_DIR_BY_DATASET = {
    "hip734_cropped": "hip",
    "Covid": "covid",
    "hip": "hip",
    "covid": "covid",
}

DATASET_NAME = {
    "hip": "hip734_cropped",
    "covid": "Covid",
    "hip734_cropped": "hip734_cropped",
    "Covid": "Covid",
}


def parse_ar_list(s: str) -> list[float]:
    return [float(p.strip()) for p in s.split(",") if p.strip()]


def load_scalar_curve(run_dir: Path, tag: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return (steps, values) for a TB scalar tag, or None if missing."""
    if not run_dir.is_dir():
        return None
    events = list(run_dir.glob("events.out.tfevents*"))
    if not events:
        return None
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None
    scalars = ea.Scalars(tag)
    if not scalars:
        return None
    # Deduplicate by step (keep last) in case of resumed runs.
    by_step: dict[int, float] = {}
    for s in scalars:
        by_step[int(s.step)] = float(s.value)
    steps = np.array(sorted(by_step), dtype=float)
    values = np.array([by_step[int(st)] for st in steps], dtype=float)
    return steps, values


def load_fold_curves(
    eval_root: Path,
    dataset: str,
    experiment: str,
    ar: float,
    tag: str = VAL_TAG,
) -> list[tuple[np.ndarray, np.ndarray]]:
    curves = []
    for k in range(K_FOLDS):
        run_dir = eval_root / dataset / experiment / f"ar{ar}" / f"k{k}"
        loaded = load_scalar_curve(run_dir, tag)
        if loaded is not None:
            curves.append(loaded)
    return curves


def aggregate_curves(
    curves: list[tuple[np.ndarray, np.ndarray]],
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Align folds on shared steps; return steps, mean, stderr."""
    if not curves:
        return None
    # Use intersection of steps so means are well-defined.
    common = set(curves[0][0].astype(int).tolist())
    for steps, _ in curves[1:]:
        common &= set(steps.astype(int).tolist())
    if not common:
        # Fall back to shortest length positional align if steps differ.
        min_len = min(len(v) for _, v in curves)
        stacked = np.stack([v[:min_len] for _, v in curves], axis=0)
        steps = curves[0][0][:min_len]
        mean = stacked.mean(axis=0)
        stderr = stacked.std(axis=0, ddof=0) / np.sqrt(stacked.shape[0])
        return steps, mean, stderr

    steps = np.array(sorted(common), dtype=float)
    rows = []
    for st, vals in curves:
        lookup = {int(s): v for s, v in zip(st, vals)}
        rows.append([lookup[int(s)] for s in steps])
    stacked = np.asarray(rows, dtype=float)
    mean = stacked.mean(axis=0)
    stderr = stacked.std(axis=0, ddof=0) / np.sqrt(stacked.shape[0])
    return steps, mean, stderr


def plot_comparison(
    sample_agg: tuple[np.ndarray, np.ndarray, np.ndarray],
    label_agg: tuple[np.ndarray, np.ndarray, np.ndarray],
    title: str,
    out_path: Path,
    ylabel: str = "Validation CE",
):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for agg, name, color in (
        (sample_agg, "Sample-type (medivision)", "#1f77b4"),
        (label_agg, "Label-type (label-boost)", "#d62728"),
    ):
        steps, mean, stderr = agg
        ax.plot(steps, mean, label=name, color=color, linewidth=1.8)
        ax.fill_between(steps, mean - stderr, mean + stderr, color=color, alpha=0.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_grid(
    panels: list[tuple[str, Optional[tuple], Optional[tuple]]],
    title: str,
    out_path: Path,
):
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (name, sample_agg, label_agg) in zip(axes, panels):
        if sample_agg is not None:
            st, m, se = sample_agg
            ax.plot(st, m, label="Sample-type", color="#1f77b4", linewidth=1.6)
            ax.fill_between(st, m - se, m + se, color="#1f77b4", alpha=0.2)
        if label_agg is not None:
            st, m, se = label_agg
            ax.plot(st, m, label="Label-type", color="#d62728", linewidth=1.6)
            ax.fill_between(st, m - se, m + se, color="#d62728", alpha=0.2)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Validation CE")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="hip", choices=list(DATASET_NAME))
    parser.add_argument(
        "--ar", default="0.1,0.5,1.0",
        help="Comma-separated annotation ratios (default: 0.1,0.5,1.0)",
    )
    parser.add_argument(
        "--experiments", default=",".join(DEFAULT_EXPERIMENTS),
        help="Comma-separated experiment names",
    )
    parser.add_argument(
        "--label-root",
        default="/Users/ricepoutine/Documents/repos.nosync/label-boost/eval_output",
        help="Label-type TB root (label-boost eval_output)",
    )
    parser.add_argument(
        "--sample-root",
        default="/Users/ricepoutine/Documents/repos.nosync/medivision/eval_output",
        help="Sample-type TB root (medivision eval_output)",
    )
    parser.add_argument(
        "--fig-root",
        default="/Users/ricepoutine/Documents/repos.nosync/label-boost/figures",
        help="Figures root (default: label-boost/figures)",
    )
    parser.add_argument(
        "--tag", default=VAL_TAG,
        help=f"TB scalar tag (default: {VAL_TAG})",
    )
    args = parser.parse_args()

    dataset = DATASET_NAME[args.dataset]
    fig_dir = Path(args.fig_root) / FIG_DIR_BY_DATASET[args.dataset] / "learning_curves"
    ar_list = parse_ar_list(args.ar)
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    label_root = Path(args.label_root)
    sample_root = Path(args.sample_root)

    print(f"Dataset: {dataset}")
    print(f"Label-type root:  {label_root}")
    print(f"Sample-type root: {sample_root}")
    print(f"Output: {fig_dir}")
    print(f"Tag: {args.tag}")

    for ar in ar_list:
        grid_panels = []
        for exp in experiments:
            sample_curves = load_fold_curves(sample_root, dataset, exp, ar, args.tag)
            label_curves = load_fold_curves(label_root, dataset, exp, ar, args.tag)
            sample_agg = aggregate_curves(sample_curves)
            label_agg = aggregate_curves(label_curves)

            display = DISPLAY_NAMES.get(exp, exp)
            print(
                f"  ar={ar} {exp}: sample_folds={len(sample_curves)} "
                f"label_folds={len(label_curves)}"
            )
            if sample_agg is None or label_agg is None:
                print(
                    f"    skip (sample={sample_agg is not None}, "
                    f"label={label_agg is not None})"
                )
                continue

            out = fig_dir / f"val_ce_{exp}_ar{ar}.png"
            plot_comparison(
                sample_agg, label_agg,
                title=f"{display}  |  ar={ar}",
                out_path=out,
            )
            print(f"    wrote {out}")
            grid_panels.append((display, sample_agg, label_agg))

        if grid_panels:
            grid_out = fig_dir / f"val_ce_grid_ar{ar}.png"
            plot_grid(
                grid_panels,
                title=f"Validation CE learning curves  |  {dataset}  |  ar={ar}",
                out_path=grid_out,
            )
            print(f"  wrote grid {grid_out}")

    print("Done.")


if __name__ == "__main__":
    main()
