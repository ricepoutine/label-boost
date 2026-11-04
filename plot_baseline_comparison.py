"""
plot_baseline_comparison.py - SSL / KD baselines vs supervised and Label-Boost.

Reads TensorBoard folds and ``summary_all_ar.csv`` from:

    curves_baseline_output/     Mean Teacher, KD (MII / ViTMAE, T=4)
    old_eval_output/            ResNet18 (validation CE and test summaries;
                                this run did not log validation top-1)
    curves_sample_type_output/  ResNet18_soft
    curves_eval_output/         Label-Boost MII and ViTMAE (label-type, λ=1.0)

Writes, per dataset::

    learning_curves/{hip,covid}/baseline_comparison/
        accuracy_vs_ar.png
        auroc_vs_ar.png
        val_top1_grid.png
        val_top1_ar{ar}.png
        val_ce_grid.png
        val_ce_ar{ar}.png

Examples::

    python plot_baseline_comparison.py
    python plot_baseline_comparison.py --dataset hip --ar 0.1,0.5,1.0
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)

from plot_learning_curves import (  # noqa: E402
    DATASET_NAME,
    FIG_DIR_BY_DATASET,
    K_FOLDS,
    REPO_ROOT,
    aggregate_curves,
    parse_ar_list,
)

VAL_TOP1 = "Valid/Top 1 accuracy"
VAL_CE = "Valid/CE Losses"
CURVE_TAGS = (VAL_TOP1, VAL_CE)
DEFAULT_ARS = [round(i * 0.1, 1) for i in range(1, 11)]
DETAIL_ARS = [0.1, 0.5, 1.0]
SMOOTH_WINDOW = 11

# name, legend, root key, color, linestyle, linewidth
SERIES: list[tuple[str, str, str, str, str, float]] = [
    ("ResNet18", "Radiologist", "old", "black", "-", 2.0),
    ("ResNet18_soft", "Regularized", "sample", "#7f7f7f", "-", 2.0),
    ("MeanTeacher", "Mean Teacher", "baseline", "#2ca02c", "--", 1.8),
    ("KD_MII_T4", "KD (MII, T=4)", "baseline", "#9467bd", "--", 1.8),
    ("KD_ViTMAE_T4", "KD (ViTMAE, T=4)", "baseline", "#ff7f0e", "--", 1.8),
    ("KD_MII_T4_2step", "KD (MII, T=4, 2-step)", "baseline", "#9467bd", "-.", 1.8),
    ("KD_ViTMAE_T4_2step", "KD (ViTMAE, T=4, 2-step)", "baseline", "#ff7f0e", "-.", 1.8),
    ("ResNet18_MII_l1.0", "Label-Boost (MII, λ=1.0)", "label", "#d62728", "-", 2.2),
    ("ResNet18_ViTMAE_l1.0", "Label-Boost (ViTMAE, λ=1.0)", "label", "#1f77b4", "-", 2.2),
]

Agg = tuple[np.ndarray, np.ndarray, np.ndarray]


def roots() -> dict[str, Path]:
    return {
        "baseline": REPO_ROOT / "curves_baseline_output",
        "sample": REPO_ROOT / "curves_sample_type_output",
        "label": REPO_ROOT / "curves_eval_output",
        "old": REPO_ROOT / "old_eval_output",
    }


def load_run_tags(
    run_dir: Path, tags: tuple[str, ...],
) -> dict[str, Optional[tuple[np.ndarray, np.ndarray]]]:
    found: dict[str, Optional[tuple[np.ndarray, np.ndarray]]] = {t: None for t in tags}
    if not run_dir.is_dir() or not list(run_dir.glob("events.out.tfevents*")):
        return found
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    for tag in tags:
        if tag not in available:
            continue
        by_step: dict[int, float] = {}
        for scalar in ea.Scalars(tag):
            by_step[int(scalar.step)] = float(scalar.value)
        if not by_step:
            continue
        steps = np.array(sorted(by_step), dtype=float)
        values = np.array([by_step[int(s)] for s in steps], dtype=float)
        found[tag] = (steps, values)
    return found


def load_experiment_curves(
    root: Path,
    dataset: str,
    experiment: str,
    ar_list: list[float],
) -> dict[float, dict[str, Optional[Agg]]]:
    """Fold-mean curves keyed by annotation ratio, then scalar tag."""
    out: dict[float, dict[str, Optional[Agg]]] = {}
    for ar in ar_list:
        by_tag: dict[str, list] = {tag: [] for tag in CURVE_TAGS}
        for k in range(K_FOLDS):
            run_dir = root / dataset / experiment / f"ar{ar}" / f"k{k}"
            loaded = load_run_tags(run_dir, CURVE_TAGS)
            for tag in CURVE_TAGS:
                if loaded[tag] is not None:
                    by_tag[tag].append(loaded[tag])
        out[ar] = {tag: aggregate_curves(curves) for tag, curves in by_tag.items()}
        n_top1 = len(by_tag[VAL_TOP1])
        n_ce = len(by_tag[VAL_CE])
        if n_top1 == 0 and n_ce == 0:
            print(f"    {experiment} ar={ar}: no curves")
        else:
            print(f"    {experiment} ar={ar}: top1_folds={n_top1} ce_folds={n_ce}")
    return out


def load_summary(root: Path, dataset: str, experiment: str) -> Optional[pd.DataFrame]:
    path = root / dataset / experiment / "summary_all_ar.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if "ar" not in df.columns:
        return None
    return df.sort_values("ar")


def _style_axis(ax: plt.Axes, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _legend_outside(ax: plt.Axes) -> None:
    ax.legend(
        frameon=False, fontsize=8,
        loc="center left", bbox_to_anchor=(1.02, 0.5),
    )


def smooth_agg(agg: Agg, window: int = SMOOTH_WINDOW) -> Agg:
    """Centered moving average so fold-mean curves can be compared."""
    steps, mean, stderr = agg
    mean_s = pd.Series(mean).rolling(window, center=True, min_periods=1).mean().to_numpy()
    err_s = pd.Series(stderr).rolling(window, center=True, min_periods=1).mean().to_numpy()
    return steps, mean_s, err_s


def plot_metric_vs_ar(
    frames: list[tuple[str, str, str, float, pd.DataFrame]],
    mean_col: str,
    err_col: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    any_drawn = False
    for display, color, ls, lw, df in frames:
        if mean_col not in df.columns:
            continue
        sub = df[["ar", mean_col, err_col]].dropna(subset=[mean_col])
        if sub.empty:
            continue
        any_drawn = True
        ax.errorbar(
            sub["ar"], sub[mean_col], yerr=sub[err_col],
            label=display, color=color, linestyle=ls, linewidth=lw,
            marker="o", markersize=5, capsize=3, elinewidth=1.0,
        )
    if not any_drawn:
        plt.close(fig)
        print(f"  skip {out_path.name} (no summary rows)")
        return
    _style_axis(ax, "Labeling ratio", ylabel)
    ax.set_title(title)
    ax.set_xticks(DEFAULT_ARS)
    _legend_outside(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def _draw_series(
    ax: plt.Axes,
    drawn: list[tuple[str, str, str, float, Agg]],
    bands: bool,
) -> None:
    for display, color, ls, lw, agg in drawn:
        steps, mean, stderr = smooth_agg(agg)
        ax.plot(steps, mean, label=display, color=color, linestyle=ls, linewidth=lw)
        if bands:
            ax.fill_between(
                steps, mean - stderr, mean + stderr,
                color=color, alpha=0.13, linewidth=0,
            )


def _ce_ylim(drawn: list[tuple[str, str, str, float, Agg]]) -> tuple[float, float]:
    """Keep the y-axis on the settled loss so the first-epoch spike does not dominate."""
    highs = []
    for _, _, _, _, agg in drawn:
        steps, mean, _ = smooth_agg(agg)
        tail = mean[steps >= 20] if np.any(steps >= 20) else mean
        highs.append(float(np.max(tail)))
    ymax = max(highs) * 1.08
    return 0.0, ymax


def plot_curve(
    drawn: list[tuple[str, str, str, float, Agg]],
    title: str,
    ylabel: str,
    out_path: Path,
    ylim: Optional[tuple[float, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    _draw_series(ax, drawn, bands=True)
    _style_axis(ax, "Epoch", ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_title(title)
    _legend_outside(ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_curve_grid(
    panels: list[tuple[float, list[tuple[str, str, str, float, Agg]]]],
    title: str,
    ylabel: str,
    out_path: Path,
    ylim_from_ce: bool = False,
) -> None:
    n = len(panels)
    cols = 5 if n > 5 else max(n, 1)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.15 * rows), sharex=True)
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (ar, drawn) in zip(axes_flat, panels):
        _draw_series(ax, drawn, bands=False)
        ax.set_title(f"ar={ar:g}", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ylim_from_ce and drawn:
            ax.set_ylim(*_ce_ylim(drawn))
    for ax in axes_flat[n:]:
        ax.axis("off")
    axes_flat[0].set_ylabel(ylabel)
    for ax in axes_flat[(rows - 1) * cols:(rows - 1) * cols + min(cols, n)]:
        if ax.axison:
            ax.set_xlabel("Epoch")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, frameon=False, fontsize=8,
            loc="upper center", ncol=min(4, len(handles)),
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.suptitle(title, fontsize=13, y=1.08)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def _drawn(
    curves: dict[str, dict[float, dict[str, Optional[Agg]]]],
    ar: float,
    tag: str,
) -> list[tuple[str, str, str, float, Agg]]:
    drawn = []
    for name, _display, _root_key, color, ls, lw in SERIES:
        agg = curves.get(name, {}).get(ar, {}).get(tag)
        if agg is None:
            continue
        display = next(s[1] for s in SERIES if s[0] == name)
        drawn.append((display, color, ls, lw, agg))
    return drawn


def run_dataset(dataset_key: str, ar_list: list[float]) -> None:
    dataset = DATASET_NAME[dataset_key]
    pretty = "Hip" if dataset_key.startswith("hip") else "Covid"
    out_dir = REPO_ROOT / "learning_curves" / FIG_DIR_BY_DATASET[dataset_key] / "baseline_comparison"
    print(f"\nDataset: {dataset}")
    print(f"Output: {out_dir}")

    root_map = roots()
    summaries: list[tuple[str, str, str, float, pd.DataFrame]] = []
    curves: dict[str, dict[float, dict[str, Optional[Agg]]]] = {}

    for name, display, root_key, color, ls, lw in SERIES:
        root = root_map[root_key]
        summary = load_summary(root, dataset, name)
        if summary is not None:
            summaries.append((display, color, ls, lw, summary))
        else:
            print(f"  no summary for {name}")
        exp_dir = root / dataset / name
        if not exp_dir.is_dir():
            print(f"  no curves directory for {name}")
            continue
        print(f"  loading {name} from {root.name}")
        curves[name] = load_experiment_curves(root, dataset, name, ar_list)

    plot_metric_vs_ar(
        summaries, "top1_mean", "top1_stderr",
        title=f"{pretty} test accuracy",
        ylabel="Accuracy",
        out_path=out_dir / "accuracy_vs_ar.png",
    )
    plot_metric_vs_ar(
        summaries, "auroc_mean", "auroc_stderr",
        title=f"{pretty} test AUROC",
        ylabel="AUROC",
        out_path=out_dir / "auroc_vs_ar.png",
    )

    top1_panels = []
    ce_panels = []
    for ar in ar_list:
        top1 = _drawn(curves, ar, VAL_TOP1)
        ce = _drawn(curves, ar, VAL_CE)
        if top1:
            top1_panels.append((ar, top1))
            if ar in DETAIL_ARS or ar in ar_list and len(ar_list) <= 3:
                plot_curve(
                    top1,
                    title=f"{pretty} validation top-1, {SMOOTH_WINDOW}-epoch mean  |  ar={ar:g}",
                    ylabel="Validation top-1",
                    out_path=out_dir / f"val_top1_ar{ar:g}.png",
                )
        if ce:
            ce_panels.append((ar, ce))
            if ar in DETAIL_ARS or len(ar_list) <= 3:
                plot_curve(
                    ce,
                    title=f"{pretty} validation CE, {SMOOTH_WINDOW}-epoch mean  |  ar={ar:g}",
                    ylabel="Validation CE",
                    out_path=out_dir / f"val_ce_ar{ar:g}.png",
                    ylim=_ce_ylim(ce),
                )

    if top1_panels:
        plot_curve_grid(
            top1_panels,
            title=f"{pretty} validation top-1, {SMOOTH_WINDOW}-epoch mean",
            ylabel="Validation top-1",
            out_path=out_dir / "val_top1_grid.png",
        )
    if ce_panels:
        plot_curve_grid(
            ce_panels,
            title=f"{pretty} validation CE, {SMOOTH_WINDOW}-epoch mean",
            ylabel="Validation CE",
            out_path=out_dir / "val_ce_grid.png",
            ylim_from_ce=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="all",
        choices=["all", "hip", "covid", "hip734_cropped", "Covid"],
    )
    parser.add_argument(
        "--ar", default=",".join(f"{ar:g}" for ar in DEFAULT_ARS),
        help="Comma-separated annotation ratios (default: 0.1 through 1.0)",
    )
    args = parser.parse_args()
    ar_list = parse_ar_list(args.ar)
    if args.dataset == "all":
        dataset_keys = ["hip", "covid"]
    elif args.dataset in ("hip", "hip734_cropped"):
        dataset_keys = ["hip"]
    else:
        dataset_keys = ["covid"]
    for key in dataset_keys:
        run_dataset(key, ar_list)
    print("Done.")


if __name__ == "__main__":
    main()
