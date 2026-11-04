"""
plot_val_top1_curves.py - Validation top-1 learning speed, sample-type vs label-type.

Reads ``Valid/Top 1 accuracy`` from the per-fold TensorBoard runs under
``curves_sample_type_output/`` (``evaluate.py``) and ``curves_eval_output/``
(``new_evaluate.py``). Folds are averaged, then the curve is kept at every
``--stride`` epoch (default 1, every epoch).

Writes::

    learning_curves/{hip,covid}/
        val_top1_{experiment}_ar{ar}.png
        val_top1_grid_ar{ar}.png
        epochs_to_target.png

``epochs_to_target.png`` is the epoch at which each update type first reaches
the lower of the two methods' final validation top-1. The smaller epoch learned
that accuracy sooner.

Examples::

    python plot_val_top1_curves.py
    python plot_val_top1_curves.py --dataset hip --ar 0.1,0.5,1.0
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plot_learning_curves import (  # noqa: E402
    DATASET_NAME,
    DISPLAY_NAMES,
    FIG_DIR_BY_DATASET,
    REPO_ROOT,
    aggregate_curves,
    load_fold_curves,
    parse_ar_list,
)

VAL_TOP1_TAG = "Valid/Top 1 accuracy"
SAMPLE_COLOR = "#1f77b4"
LABEL_COLOR = "#d62728"
SAMPLE_NAME = "Sample-type"
LABEL_NAME = "Label-type"

DEFAULT_ARS = [round(i * 0.1, 1) for i in range(1, 11)]


def subsample(
    agg: tuple[np.ndarray, np.ndarray, np.ndarray],
    stride: int,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    steps, mean, stderr = agg
    mask = (steps.astype(int) % stride) == 0
    if not np.any(mask):
        return None
    return steps[mask], mean[mask], stderr[mask]


def first_epoch_at(
    steps: np.ndarray, mean: np.ndarray, target: float,
) -> Optional[float]:
    hit = np.where(mean >= target)[0]
    if hit.size == 0:
        return None
    return float(steps[hit[0]])


def discover_shared_experiments(sample_root: Path, label_root: Path, dataset: str) -> list[str]:
    sample_dir = sample_root / dataset
    label_dir = label_root / dataset
    if not sample_dir.is_dir() or not label_dir.is_dir():
        return []
    sample_names = {p.name for p in sample_dir.iterdir() if p.is_dir()}
    label_names = {p.name for p in label_dir.iterdir() if p.is_dir()}
    return sorted(sample_names & label_names)


def plot_comparison(
    sample_agg: tuple[np.ndarray, np.ndarray, np.ndarray],
    label_agg: tuple[np.ndarray, np.ndarray, np.ndarray],
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for agg, name, color in (
        (sample_agg, SAMPLE_NAME, SAMPLE_COLOR),
        (label_agg, LABEL_NAME, LABEL_COLOR),
    ):
        steps, mean, stderr = agg
        ax.plot(steps, mean, label=name, color=color, linewidth=1.8)
        ax.fill_between(steps, mean - stderr, mean + stderr, color=color, alpha=0.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation top-1")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_grid(
    panels: list[tuple[str, tuple, tuple]],
    title: str,
    out_path: Path,
) -> None:
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (name, sample_agg, label_agg) in zip(axes, panels):
        for agg, series, color in (
            (sample_agg, SAMPLE_NAME, SAMPLE_COLOR),
            (label_agg, LABEL_NAME, LABEL_COLOR),
        ):
            steps, mean, stderr = agg
            ax.plot(steps, mean, label=series, color=color, linewidth=1.6)
            ax.fill_between(steps, mean - stderr, mean + stderr, color=color, alpha=0.2)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Validation top-1")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_epochs_to_target(
    rows: list[dict],
    title: str,
    out_path: Path,
) -> None:
    """Grouped bars: epoch at which each method first hits the shared target."""
    experiments = []
    for row in rows:
        if row["experiment"] not in experiments:
            experiments.append(row["experiment"])
    n = len(experiments)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, exp in zip(axes, experiments):
        exp_rows = [r for r in rows if r["experiment"] == exp]
        ars = [r["ar"] for r in exp_rows]
        x = np.arange(len(ars), dtype=float)
        width = 0.36
        sample_epochs = [r["sample_epoch"] for r in exp_rows]
        label_epochs = [r["label_epoch"] for r in exp_rows]
        ax.bar(x - width / 2, sample_epochs, width, label=SAMPLE_NAME, color=SAMPLE_COLOR)
        ax.bar(x + width / 2, label_epochs, width, label=LABEL_NAME, color=LABEL_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{ar:g}" for ar in ars], rotation=0)
        ax.set_xlabel("Labeling ratio")
        ax.set_title(DISPLAY_NAMES.get(exp, exp), fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Epoch to shared target")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_dataset(
    dataset_key: str,
    ar_list: list[float],
    experiments: Optional[list[str]],
    sample_root: Path,
    label_root: Path,
    fig_root: Path,
    stride: int,
) -> None:
    dataset = DATASET_NAME[dataset_key]
    fig_dir = fig_root / FIG_DIR_BY_DATASET[dataset_key]
    if experiments is None:
        experiments = discover_shared_experiments(sample_root, label_root, dataset)
    print(f"Dataset: {dataset}")
    print(f"Experiments: {experiments}")
    print(f"Output: {fig_dir}")

    speed_rows: list[dict] = []
    for ar in ar_list:
        grid_panels = []
        for exp in experiments:
            sample_curves = load_fold_curves(sample_root, dataset, exp, ar, VAL_TOP1_TAG)
            label_curves = load_fold_curves(label_root, dataset, exp, ar, VAL_TOP1_TAG)
            sample_agg = aggregate_curves(sample_curves)
            label_agg = aggregate_curves(label_curves)
            display = DISPLAY_NAMES.get(exp, exp)
            print(
                f"  ar={ar} {exp}: sample_folds={len(sample_curves)} "
                f"label_folds={len(label_curves)}"
            )
            if sample_agg is None or label_agg is None:
                print("    skip (missing curves)")
                continue
            sample_sub = subsample(sample_agg, stride)
            label_sub = subsample(label_agg, stride)
            if sample_sub is None or label_sub is None:
                print(f"    skip (no epochs divisible by {stride})")
                continue

            out = fig_dir / f"val_top1_{exp}_ar{ar}.png"
            plot_comparison(
                sample_sub, label_sub,
                title=f"{display}  |  ar={ar}",
                out_path=out,
            )
            print(f"    wrote {out}")
            grid_panels.append((display, sample_sub, label_sub))

            target = min(float(sample_sub[1][-1]), float(label_sub[1][-1]))
            sample_epoch = first_epoch_at(sample_sub[0], sample_sub[1], target)
            label_epoch = first_epoch_at(label_sub[0], label_sub[1], target)
            if sample_epoch is None or label_epoch is None:
                print(f"    target {target:.4f} not reached")
                continue
            faster = SAMPLE_NAME if sample_epoch < label_epoch else (
                LABEL_NAME if label_epoch < sample_epoch else "tie"
            )
            print(
                f"    target={target:.4f}  sample@{sample_epoch:.0f}  "
                f"label@{label_epoch:.0f}  faster={faster}"
            )
            speed_rows.append({
                "experiment": exp,
                "ar": ar,
                "target": target,
                "sample_epoch": sample_epoch,
                "label_epoch": label_epoch,
            })

        if grid_panels:
            grid_out = fig_dir / f"val_top1_grid_ar{ar}.png"
            plot_grid(
                grid_panels,
                title=f"Validation top-1  |  {dataset}  |  ar={ar}",
                out_path=grid_out,
            )
            print(f"  wrote grid {grid_out}")

    if speed_rows:
        summary_out = fig_dir / "epochs_to_target.png"
        plot_epochs_to_target(
            speed_rows,
            title=(
                f"Epoch to reach the lower final validation top-1"
                f"  |  {dataset}"
            ),
            out_path=summary_out,
        )
        print(f"  wrote speed summary {summary_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="all",
        choices=["all", *DATASET_NAME.keys()],
        help="Dataset preset, or 'all' (default)",
    )
    parser.add_argument(
        "--ar", default=",".join(f"{ar:g}" for ar in DEFAULT_ARS),
        help="Comma-separated annotation ratios (default: 0.1 through 1.0)",
    )
    parser.add_argument(
        "--experiments", default=None,
        help="Comma-separated experiment names (default: those present in both roots)",
    )
    parser.add_argument(
        "--sample-root",
        default=str(REPO_ROOT / "curves_sample_type_output"),
    )
    parser.add_argument(
        "--label-root",
        default=str(REPO_ROOT / "curves_eval_output"),
    )
    parser.add_argument(
        "--fig-root",
        default=str(REPO_ROOT / "learning_curves"),
        help="Output root (default: learning_curves)",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Keep every Nth epoch (default: 1, every epoch)",
    )
    args = parser.parse_args()

    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    datasets = [k for k in ("hip", "covid") if k in DATASET_NAME] if args.dataset == "all" else [args.dataset]
    experiments = None
    if args.experiments:
        experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    ar_list = parse_ar_list(args.ar)

    for dataset_key in datasets:
        run_dataset(
            dataset_key,
            ar_list,
            experiments,
            Path(args.sample_root),
            Path(args.label_root),
            Path(args.fig_root),
            args.stride,
        )
    print("Done.")


if __name__ == "__main__":
    main()
