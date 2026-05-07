"""
generate_results.py - Single entry point for every paper figure.

Reads only what ``evaluate.py`` writes (per-fold predictions JSON, per-AR
``summary.csv`` rows, and the cross-AR ``summary_all_ar.csv`` aggregates
under ``eval_output/``) and reproduces the thesis figures::

    figures/{covid,hip}/{baseline,mii_l1,vitmae_l1,mii_lambdas,
                          vitmae_lambdas,mii_ablation,vitmae_ablation,
                          all_l1,all}/{accuracy,auroc}.png
    figures/{covid,hip}/pseudolabels/top1_pslabels.png
    figures/covid/calibration_mii_vs_baseline_grid.png
    figures/covid/dca_mii_vs_baseline_grid.png
    figures/umap/{mii_untrained,vitmae_base,vitmae_finetuned}_umap.png

Calibration / DCA grids re-implement the per-axes drawing using the same
LOWESS / threshold-sweep logic as ``phair_eval_suite.core_eval_functions``
(no edits to that package).

Examples::

    python generate_results.py --all
    python generate_results.py --dataset hip --preset mii_l1
    python generate_results.py --dataset covid --pseudolabels
    python generate_results.py --dataset covid --calibration-grid --dca-grid
    python generate_results.py --umap
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402
from statsmodels.nonparametric.smoothers_lowess import lowess  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (lifted from eval_results.py — the cleaned-up styling)
# ---------------------------------------------------------------------------

K_FOLDS = 5
ARS = [round(i * 0.1, 1) for i in range(1, 11)]

# Each entry mirrors evaluate.EXPERIMENTS so the same ``--experiment`` strings
# work end-to-end.
DISPLAY_NAMES: dict[str, str] = {
    "ResNet18":                       "Radiologist",
    "ResNet18_soft":                  "Regularized",
    "ResNet18_ViTMAE_l0.5":          "Label-Boost (ViTMAE, \u03bb=0.5)",
    "ResNet18_ViTMAE_l1.0":          "Label-Boost (ViTMAE, \u03bb=1.0)",
    "ResNet18_ViTMAE_l1.5":          "Label-Boost (ViTMAE, \u03bb=1.5)",
    "ResNet18_ViTMAE_l1.0_ablation": "Label-Boost (ViTMAE, \u03bb=1.0, ablation)",
    "ResNet18_MII_l0.5":             "Label-Boost (MII, \u03bb=0.5)",
    "ResNet18_MII_l1.0":             "Label-Boost (MII, \u03bb=1.0)",
    "ResNet18_MII_l1.5":             "Label-Boost (MII, \u03bb=1.5)",
    "ResNet18_MII_l1.0_ablation":    "Label-Boost (MII, \u03bb=1.0, ablation)",
}

FIXED_COLORS: dict[str, str] = {
    "ResNet18":      "black",
    "ResNet18_soft": "grey",
}

LABEL_BOOST_PALETTE = [
    "red", "blue", "orange", "green", "purple", "brown", "pink",
]

METRIC_CFG: dict[str, dict[str, str]] = {
    "top1":  {"col_mean": "top1_mean",  "col_err": "top1_stderr",
              "title": "Accuracy vs Labeling Ratio",  "yaxis": "Accuracy",
              "fname": "accuracy.png"},
    "auroc": {"col_mean": "auroc_mean", "col_err": "auroc_stderr",
              "title": "AUROC vs Labeling Ratio",     "yaxis": "AUROC",
              "fname": "auroc.png"},
    "brier": {"col_mean": "brier_score_mean", "col_err": "brier_score_stderr",
              "title": "Brier score vs Labeling Ratio", "yaxis": "Brier score",
              "fname": "brier_score.png"},
    "calibration_slope": {
        "col_mean": "calibration_slope_mean",
        "col_err": "calibration_slope_stderr",
        "title": "Calibration slope vs Labeling Ratio",
        "yaxis": "Calibration slope",
        "fname": "calibration_slope.png",
    },
}

OPTIONAL_SUMMARY_COLS = (
    "brier_score_mean",
    "brier_score_stderr",
    "calibration_slope_mean",
    "calibration_slope_stderr",
    "pseudolabel_acc_mean",
    "pseudolabel_acc_stderr",
)

# All experiments grouped into the named comparisons used for paper figures.
PRESETS: dict[str, list[str]] = {
    "baseline":         ["ResNet18", "ResNet18_soft"],

    "vitmae_l1":        ["ResNet18", "ResNet18_soft",
                         "ResNet18_ViTMAE_l1.0"],
    "vitmae_lambdas":   ["ResNet18", "ResNet18_soft",
                         "ResNet18_ViTMAE_l0.5", "ResNet18_ViTMAE_l1.0",
                         "ResNet18_ViTMAE_l1.5"],
    "vitmae_ablation":  ["ResNet18", "ResNet18_soft",
                         "ResNet18_ViTMAE_l1.0", "ResNet18_ViTMAE_l1.0_ablation"],

    "mii_l1":           ["ResNet18", "ResNet18_soft",
                         "ResNet18_MII_l1.0"],
    "mii_lambdas":      ["ResNet18", "ResNet18_soft",
                         "ResNet18_MII_l0.5", "ResNet18_MII_l1.0",
                         "ResNet18_MII_l1.5"],
    "mii_ablation":     ["ResNet18", "ResNet18_soft",
                         "ResNet18_MII_l1.0", "ResNet18_MII_l1.0_ablation"],

    "all_l1":           ["ResNet18", "ResNet18_soft",
                         "ResNet18_ViTMAE_l1.0", "ResNet18_MII_l1.0"],

    # NEW: every experiment side-by-side (mirrors figures/{dataset}/all/).
    "all":              [
        "ResNet18", "ResNet18_soft",
        "ResNet18_ViTMAE_l0.5", "ResNet18_ViTMAE_l1.0", "ResNet18_ViTMAE_l1.5",
        "ResNet18_ViTMAE_l1.0_ablation",
        "ResNet18_MII_l0.5", "ResNet18_MII_l1.0", "ResNet18_MII_l1.5",
        "ResNet18_MII_l1.0_ablation",
    ],
}

# Datasets recognized by --dataset / --all paths.  Keys here match the
# ``preset.dataset_name`` written by evaluate.py (so eval_output paths work).
DATASETS: tuple[str, ...] = ("hip734_cropped", "Covid")

# Per-figure-folder name normalization: when invoking --dataset hip we want
# figures to land under figures/hip/, but eval_output uses hip734_cropped.
FIG_DIR_BY_DATASET: dict[str, str] = {
    "hip734_cropped": "hip",
    "Covid":          "covid",
}


# ---------------------------------------------------------------------------
# Data loading (lifted with minor pruning from eval_results.py)
# ---------------------------------------------------------------------------

def discover_experiments(eval_root: str, dataset: str) -> list[str]:
    base = Path(eval_root) / dataset
    if not base.is_dir():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        has_aggregate = (d / "summary_all_ar.csv").is_file()
        has_any_ar = any((d / f"ar{ar}" / "summary.csv").is_file() for ar in ARS)
        if has_aggregate or has_any_ar:
            out.append(d.name)
    return out


def _ensure_optional_cols(df: pd.DataFrame) -> pd.DataFrame:
    for c in OPTIONAL_SUMMARY_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df


def load_summary(eval_root: str, dataset: str, experiment: str) -> pd.DataFrame:
    """Load cross-AR summary; falls back to per-AR ``summary.csv`` files."""
    agg_path = Path(eval_root) / dataset / experiment / "summary_all_ar.csv"
    if agg_path.is_file():
        return _ensure_optional_cols(pd.read_csv(agg_path))

    exp_dir = Path(eval_root) / dataset / experiment
    rows: list[dict] = []
    for ar in ARS:
        ar_csv = exp_dir / f"ar{ar}" / "summary.csv"
        if not ar_csv.is_file():
            continue
        df = pd.read_csv(ar_csv)
        mean_df = df[df["k"] == "mean"]
        std_df = df[df["k"] == "std"]
        if mean_df.empty or std_df.empty:
            fold_rows = df[~df["k"].isin(["mean", "std"])]
            top1s = fold_rows["top1"].astype(float).values
            aurocs = fold_rows["auroc"].astype(float).values
            n = len(top1s)
            rows.append({
                "ar": ar,
                "top1_mean": np.mean(top1s),
                "top1_stderr": np.std(top1s) / np.sqrt(n),
                "auroc_mean": np.mean(aurocs),
                "auroc_stderr": np.std(aurocs) / np.sqrt(n),
            })
            continue

        mrow = mean_df.iloc[0]
        srow = std_df.iloc[0]
        rec: dict = {
            "ar": ar,
            "top1_mean": float(mrow["top1"]),
            "top1_stderr": float(srow["top1"]),
            "auroc_mean": float(mrow["auroc"]),
            "auroc_stderr": float(srow["auroc"]),
        }
        for opt_col in ("pseudolabel_acc", "brier_score", "calibration_slope"):
            if opt_col in mrow.index and pd.notna(mrow.get(opt_col)):
                rec[f"{opt_col}_mean"] = float(mrow[opt_col])
                rec[f"{opt_col}_stderr"] = float(srow[opt_col])
        rows.append(rec)

    if not rows:
        raise FileNotFoundError(f"No summary data for {dataset}/{experiment}")
    return _ensure_optional_cols(pd.DataFrame(rows))


def load_all_summaries(
    eval_root: str, dataset: str, experiments: list[str],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for exp in experiments:
        try:
            out[exp] = load_summary(eval_root, dataset, exp)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} - skipping")
    return out


def load_fold_table(
    eval_root: str, dataset: str, experiment: str, ar: float,
) -> Optional[pd.DataFrame]:
    p = Path(eval_root) / dataset / experiment / f"ar{ar}" / "summary.csv"
    if not p.is_file():
        return None
    df = pd.read_csv(p)
    return df[~df["k"].isin(["mean", "std"])].copy()


def load_pooled_predictions(
    eval_root: str, dataset: str, experiment: str, ar: float,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Concatenate fold_*_predictions.json -> (y_true, y_prob) or None."""
    base = Path(eval_root) / dataset / experiment / f"ar{ar}"
    files = sorted(base.glob("fold_*_predictions.json"))
    if not files:
        return None
    y_true: list[float] = []
    y_prob: list[float] = []
    for f in files:
        with open(f, encoding="utf-8") as h:
            d = json.load(h)
        y_true.extend(d["y_true"])
        y_prob.extend(d.get("y_proba", d.get("y_prob", [])))
    return np.asarray(y_true), np.asarray(y_prob)


# ---------------------------------------------------------------------------
# Statistical significance (paired t-test vs first preset experiment)
# ---------------------------------------------------------------------------

def _load_fold_values(
    eval_root: str, dataset: str, experiment: str, ar: float, metric: str,
) -> Optional[np.ndarray]:
    df = load_fold_table(eval_root, dataset, experiment, ar)
    if df is None or metric not in df.columns:
        return None
    return df[metric].astype(float).values


def compute_significance(
    eval_root: str, dataset: str, baseline: str,
    experiments: list[str], metrics: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    for exp in experiments:
        if exp == baseline:
            continue
        for ar in ARS:
            for metric in metrics:
                col = METRIC_CFG[metric]["col_mean"].replace("_mean", "")
                base_vals = _load_fold_values(eval_root, dataset, baseline, ar, col)
                exp_vals = _load_fold_values(eval_root, dataset, exp, ar, col)
                if base_vals is None or exp_vals is None:
                    continue
                if len(base_vals) != len(exp_vals):
                    continue

                t_stat, p_value = stats.ttest_rel(exp_vals, base_vals)
                diff = exp_vals - base_vals
                if p_value < 0.001:
                    sig = "***"
                elif p_value < 0.01:
                    sig = "**"
                elif p_value < 0.05:
                    sig = "*"
                else:
                    sig = ""
                rows.append({
                    "experiment": exp,
                    "display_name": DISPLAY_NAMES.get(exp, exp),
                    "baseline": baseline,
                    "ar": ar,
                    "metric": METRIC_CFG[metric]["yaxis"],
                    "baseline_mean": float(np.mean(base_vals)),
                    "experiment_mean": float(np.mean(exp_vals)),
                    "diff_mean": float(np.mean(diff)),
                    "t_stat": float(t_stat),
                    "p_value": float(p_value),
                    "significance": sig,
                })
    return pd.DataFrame(rows)


def write_significance_tables(sig_df: pd.DataFrame, out_dir: Path) -> None:
    if sig_df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    sig_df.to_csv(out_dir / "significance.csv", index=False)
    print(f"  Saved significance: {out_dir / 'significance.csv'}")

    md = [
        "# Statistical Significance (paired t-test vs Radiologist baseline)",
        "",
        "Significance levels: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001",
        "",
    ]
    for (exp, metric), grp in sig_df.groupby(["display_name", "metric"], sort=False):
        md.append(f"### {exp} \u2014 {metric}")
        md.append("")
        md.append("| AR | Baseline | Experiment | Diff | t | p | Sig |")
        md.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, row in grp.iterrows():
            md.append(
                f"| {row['ar']:.1f} "
                f"| {row['baseline_mean']:.4f} "
                f"| {row['experiment_mean']:.4f} "
                f"| {row['diff_mean']:+.4f} "
                f"| {row['t_stat']:.3f} "
                f"| {row['p_value']:.4f} "
                f"| {row['significance']} |"
            )
        md.append("")
    (out_dir / "significance.md").write_text("\n".join(md) + "\n")
    print(f"  Saved significance: {out_dir / 'significance.md'}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def write_tables(
    eval_root: str, dataset: str, experiments: list[str],
    out_dir: Path, metrics: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    wide_rows: list[dict] = []
    md_sections: list[str] = []

    for exp in experiments:
        try:
            df = load_summary(eval_root, dataset, exp)
        except FileNotFoundError as e:
            print(f"  WARNING: {e} - skipping table for {exp}")
            continue
        display = DISPLAY_NAMES.get(exp, exp)

        hdr = ["AR"]
        for m in metrics:
            hdr += [f"{METRIC_CFG[m]['yaxis']} (mean)",
                    f"{METRIC_CFG[m]['yaxis']} (SE)"]
        sep = "| " + " | ".join(["---"] * len(hdr)) + " |"
        lines = [f"### {display} (`{exp}`)", "", "| " + " | ".join(hdr) + " |", sep]

        for _, row in df.iterrows():
            parts = [f"{row['ar']:.1f}"]
            row_data = {"experiment": exp, "display_name": display, "ar": row["ar"]}
            for m in metrics:
                cfg = METRIC_CFG[m]
                if cfg["col_mean"] not in row.index or pd.isna(row[cfg["col_mean"]]):
                    parts += ["\u2014", "\u2014"]
                    row_data[f"{m}_mean"] = float("nan")
                    row_data[f"{m}_stderr"] = float("nan")
                    continue
                mean_v = row[cfg["col_mean"]]
                err_v = row[cfg["col_err"]]
                parts += [f"{mean_v:.4f}", f"{err_v:.4f}"]
                row_data[f"{m}_mean"] = mean_v
                row_data[f"{m}_stderr"] = err_v
            lines.append("| " + " | ".join(parts) + " |")
            wide_rows.append(row_data)
        md_sections.append("\n".join(lines))

    (out_dir / "tables.md").write_text("\n\n".join(md_sections) + "\n")
    pd.DataFrame(wide_rows).to_csv(out_dir / "tables.csv", index=False)
    print(f"  Saved table:  {out_dir / 'tables.md'}")
    print(f"  Saved table:  {out_dir / 'tables.csv'}")


# ---------------------------------------------------------------------------
# Y-axis range and color assignment
# ---------------------------------------------------------------------------

def compute_unified_yrange(
    summaries: dict[str, pd.DataFrame], metrics: list[str],
    padding_frac: float = 0.05,
) -> tuple[float, float]:
    lo = float("inf")
    hi = float("-inf")
    for m in metrics:
        cfg = METRIC_CFG[m]
        for df in summaries.values():
            if cfg["col_mean"] not in df.columns:
                continue
            lo = min(lo, (df[cfg["col_mean"]] - df[cfg["col_err"]]).min())
            hi = max(hi, (df[cfg["col_mean"]] + df[cfg["col_err"]]).max())
    if not math.isfinite(lo) or not math.isfinite(hi):
        return (0.0, 1.0)
    span = hi - lo
    lo -= padding_frac * span
    hi += padding_frac * span
    if hi > 0.98 and all(m in ("top1", "auroc") for m in metrics):
        hi = 1.0
    lo = math.floor(lo * 100) / 100
    return lo, hi


def assign_colors(experiments: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    palette_idx = 0
    for exp in experiments:
        if exp in FIXED_COLORS:
            out[exp] = FIXED_COLORS[exp]
        else:
            out[exp] = LABEL_BOOST_PALETTE[palette_idx % len(LABEL_BOOST_PALETTE)]
            palette_idx += 1
    return out


# ---------------------------------------------------------------------------
# Plotting (scatter with error bars, plotly -> PNG)
# ---------------------------------------------------------------------------

def plot_metric_vs_ar(
    summaries: dict[str, pd.DataFrame],
    metric_key: str,
    yrange: tuple[float, float],
    out_path: Path,
    width: int = 1000,
    height: int = 700,
) -> None:
    cfg = METRIC_CFG[metric_key]
    colors = assign_colors(summaries.keys())

    fig = go.Figure()
    for exp, df in summaries.items():
        if cfg["col_mean"] not in df.columns:
            continue
        sub = df[["ar", cfg["col_mean"], cfg["col_err"]]].dropna(
            subset=[cfg["col_mean"]],
        )
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["ar"], y=sub[cfg["col_mean"]],
            mode="markers",
            name=DISPLAY_NAMES.get(exp, exp),
            marker=dict(size=10, color=colors[exp]),
            error_y=dict(type="data", array=sub[cfg["col_err"]], visible=True),
        ))

    if not fig.data:
        print(f"  WARNING: no finite data for {metric_key} -> skip {out_path}")
        return

    fig.update_layout(
        title=cfg["title"],
        xaxis_title="Labeling Ratio",
        yaxis_title=cfg["yaxis"],
        yaxis_range=list(yrange),
        width=width,
        height=height,
        showlegend=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path))
    print(f"  Saved figure: {out_path}")


def plot_pseudolabels(
    summaries: dict[str, pd.DataFrame],
    out_path: Path,
    title: str,
    yrange: tuple[float, float] = (0.6, 1.0),
    width: int = 1000,
    height: int = 700,
) -> None:
    """Scatter of teacher pseudolabel accuracy vs labeling ratio.

    Reads ``pseudolabel_acc_mean`` / ``pseudolabel_acc_stderr`` columns
    written by evaluate.run_experiment for any experiment that has a teacher.
    Lines are colour-coded by teacher (blue=ViTMAE, red=MII).
    """
    teacher_color = {"ViTMAE": "blue", "MII": "red"}
    teacher_data: dict[str, pd.DataFrame] = {}
    for exp, df in summaries.items():
        if exp not in DISPLAY_NAMES:
            continue
        if "pseudolabel_acc_mean" not in df.columns:
            continue
        sub = df[["ar", "pseudolabel_acc_mean", "pseudolabel_acc_stderr"]].dropna(
            subset=["pseudolabel_acc_mean"],
        )
        if sub.empty:
            continue
        teacher = "ViTMAE" if "ViTMAE" in exp else ("MII" if "MII" in exp else None)
        if teacher is None:
            continue
        # Average across experiments that share the same teacher (same teacher
        # produces the same pseudolabels regardless of student-side hyperparams,
        # so the values should already match within numerical noise; we just
        # take the first non-empty source per teacher).
        teacher_data.setdefault(teacher, sub)

    if not teacher_data:
        print(f"  WARNING: no pseudolabel_acc columns -> skip {out_path}")
        return

    fig = go.Figure()
    for teacher, sub in teacher_data.items():
        fig.add_trace(go.Scatter(
            x=sub["ar"],
            y=sub["pseudolabel_acc_mean"],
            mode="markers",
            name=teacher,
            marker=dict(size=10, color=teacher_color.get(teacher, "grey")),
            error_y=dict(type="data", array=sub["pseudolabel_acc_stderr"],
                         visible=True),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Labeling Ratio",
        yaxis_title="Pseudolabel Accuracy",
        yaxis_range=list(yrange),
        width=width,
        height=height,
        showlegend=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path))
    print(f"  Saved figure: {out_path}")


# ---------------------------------------------------------------------------
# Calibration / DCA grid (logic mirrors phair_eval_suite.core_eval_functions
# so the un-edited package stays the source of truth, but we draw onto
# subplots for the multi-AR grids)
# ---------------------------------------------------------------------------

def _calibration_on_axes(ax, y_true: np.ndarray, y_prob: np.ndarray,
                         label: str, color: str) -> tuple[float, float, float]:
    """LOWESS-smoothed calibration curve on ``ax``.

    Returns (slope, intercept, brier).  Logic matches
    ``phair_eval_suite.core_eval_functions.calibration``.
    """
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    logit_pred = np.log(y_prob_clipped / (1 - y_prob_clipped))
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(logit_pred.reshape(-1, 1), y_true)
    slope = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])
    brier = float(brier_score_loss(y_true, y_prob))

    sort_idx = np.argsort(y_prob)
    smoothed = lowess(
        y_true[sort_idx].astype(float),
        y_prob[sort_idx],
        frac=0.25, it=0, return_sorted=True,
    )
    ax.plot(smoothed[:, 0], smoothed[:, 1], linewidth=2.0, color=color,
            label=f"{label} (slope={slope:.2f})")
    return slope, intercept, brier


def _dca_on_axes(ax, y_true: np.ndarray, y_prob: np.ndarray,
                 label: str, color: str,
                 thresh_range: tuple[float, float] = (0.0, 0.5)) -> None:
    """Decision-curve analysis on ``ax`` (matches phair core_eval_functions)."""
    thresholds = np.linspace(thresh_range[0], thresh_range[1], 100)
    n = len(y_true)
    net_benefits = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        net_benefits.append(tp / n - (fp / n) * (t / (1 - t)))
    ax.plot(thresholds, net_benefits, linewidth=2.0, color=color, label=label)


def _add_calibration_reference(ax) -> None:
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Perfect")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)


def _add_dca_reference(ax, y_true: np.ndarray,
                       thresh_range: tuple[float, float] = (0.0, 0.5)) -> None:
    thresholds = np.linspace(thresh_range[0], thresh_range[1], 100)
    prevalence = float(np.mean(y_true)) if len(y_true) else 0.0
    treat_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
    ax.plot(thresholds, treat_all, "--", linewidth=1.5, alpha=0.8,
            color="#777777", label="Treat all")
    ax.axhline(0, linestyle=":", color="grey", linewidth=1.5,
               label="Treat none")
    ax.grid(alpha=0.3)
    ax.set_xlim(thresh_range[0], thresh_range[1])


def plot_calibration_grid(
    eval_root: str, dataset: str, experiments: list[str],
    out_path: Path, ars: list[float] = ARS,
) -> None:
    """Calibration grid (one panel per AR), comparing the experiments overlaid."""
    n = len(ars)
    cols = min(5, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows),
                             squeeze=False)

    colors = assign_colors(experiments)
    legend_handles = None

    for idx, ar in enumerate(ars):
        ax = axes[idx // cols][idx % cols]
        any_drawn = False
        for exp in experiments:
            data = load_pooled_predictions(eval_root, dataset, exp, ar)
            if data is None:
                continue
            y_true, y_prob = data
            if len(y_true) == 0:
                continue
            _calibration_on_axes(
                ax, y_true, y_prob,
                label=DISPLAY_NAMES.get(exp, exp),
                color=colors[exp],
            )
            any_drawn = True
        _add_calibration_reference(ax)
        ax.set_title(f"AR = {ar}", fontsize=11)
        if idx % cols == 0:
            ax.set_ylabel("Observed proportion")
        if idx // cols == rows - 1:
            ax.set_xlabel("Predicted probability")
        if any_drawn and legend_handles is None:
            legend_handles = ax.get_legend_handles_labels()

    # Hide any unused panels
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    if legend_handles:
        fig.legend(*legend_handles, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02),
                   ncol=min(4, len(legend_handles[0])), fontsize=10)

    fig.suptitle("Calibration curves (per AR)", fontsize=14, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_path}")


def plot_dca_grid(
    eval_root: str, dataset: str, experiments: list[str],
    out_path: Path, ars: list[float] = ARS,
) -> None:
    n = len(ars)
    cols = min(5, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows),
                             squeeze=False)

    colors = assign_colors(experiments)
    legend_handles = None

    for idx, ar in enumerate(ars):
        ax = axes[idx // cols][idx % cols]
        y_true_for_ref = np.array([])
        any_drawn = False
        for exp in experiments:
            data = load_pooled_predictions(eval_root, dataset, exp, ar)
            if data is None:
                continue
            y_true, y_prob = data
            if len(y_true) == 0:
                continue
            _dca_on_axes(
                ax, y_true, y_prob,
                label=DISPLAY_NAMES.get(exp, exp),
                color=colors[exp],
            )
            y_true_for_ref = y_true
            any_drawn = True
        _add_dca_reference(ax, y_true_for_ref)
        ax.set_title(f"AR = {ar}", fontsize=11)
        if idx % cols == 0:
            ax.set_ylabel("Net benefit")
        if idx // cols == rows - 1:
            ax.set_xlabel("Decision threshold")
        if any_drawn and legend_handles is None:
            legend_handles = ax.get_legend_handles_labels()

    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")

    if legend_handles:
        fig.legend(*legend_handles, loc="lower center",
                   bbox_to_anchor=(0.5, -0.02),
                   ncol=min(4, len(legend_handles[0])), fontsize=10)

    fig.suptitle("Decision-curve analysis (per AR)", fontsize=14, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: {out_path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def fig_dataset_dir(fig_root: Path, dataset: str) -> Path:
    """``figures/{covid|hip}`` regardless of the underlying dataset name."""
    name = FIG_DIR_BY_DATASET.get(dataset, dataset)
    return fig_root / name


def run_preset(
    eval_root: str,
    dataset: str,
    preset_name: str,
    experiments: list[str],
    fig_root: Path,
    metrics: list[str] | None = None,
    yrange: Optional[tuple[float, float]] = None,
    width: int = 1000,
    height: int = 700,
    write_extra: bool = True,
) -> None:
    if metrics is None:
        metrics = ["top1", "auroc"]
    out_dir = fig_dataset_dir(fig_root, dataset) / preset_name

    print(f"\n{'=' * 60}")
    print(f"Preset    : {preset_name}")
    print(f"Dataset   : {dataset}")
    print(f"Experiments: {experiments}")
    print(f"Output    : {out_dir}")
    print(f"{'=' * 60}")

    summaries = load_all_summaries(eval_root, dataset, experiments)
    if not summaries:
        print("  No data found - nothing to do.")
        return
    if yrange is None:
        yrange = compute_unified_yrange(summaries, metrics)
    print(f"  Y-axis range: [{yrange[0]:.3f}, {yrange[1]:.3f}]")

    for m in metrics:
        plot_metric_vs_ar(summaries, m, yrange,
                          out_dir / METRIC_CFG[m]["fname"], width, height)

    if write_extra:
        write_tables(eval_root, dataset, list(summaries.keys()), out_dir, metrics)
        baseline = experiments[0]
        sig_df = compute_significance(eval_root, dataset, baseline,
                                      list(summaries.keys()), metrics)
        write_significance_tables(sig_df, out_dir)


def run_pseudolabels(
    eval_root: str, dataset: str, fig_root: Path,
    width: int = 1000, height: int = 700,
) -> None:
    teacher_experiments = [
        "ResNet18_ViTMAE_l1.0", "ResNet18_MII_l1.0",
    ]
    summaries = load_all_summaries(eval_root, dataset, teacher_experiments)
    if not summaries:
        print(f"  No teacher data for pseudolabel plot ({dataset})")
        return
    out_path = fig_dataset_dir(fig_root, dataset) / "pseudolabels" / "top1_pslabels.png"
    pretty_dataset = "Hip" if dataset.startswith("hip") else dataset
    yrange = (0.89, 1.0) if dataset == "Covid" else (0.6, 1.0)
    plot_pseudolabels(summaries, out_path,
                      title=f"Pseudolabel Accuracy vs Labeling Ratio ({pretty_dataset})",
                      yrange=yrange, width=width, height=height)


def run_calibration_grid(
    eval_root: str, dataset: str, fig_root: Path,
    experiments: tuple[str, str] = ("ResNet18", "ResNet18_MII_l1.0"),
) -> None:
    out_path = fig_dataset_dir(fig_root, dataset) / "calibration_mii_vs_baseline_grid.png"
    plot_calibration_grid(eval_root, dataset, list(experiments), out_path)


def run_dca_grid(
    eval_root: str, dataset: str, fig_root: Path,
    experiments: tuple[str, str] = ("ResNet18", "ResNet18_MII_l1.0"),
) -> None:
    out_path = fig_dataset_dir(fig_root, dataset) / "dca_mii_vs_baseline_grid.png"
    plot_dca_grid(eval_root, dataset, list(experiments), out_path)


def run_umap(fig_root: Path, dataset: str = "hip", k: int = 0, ar: float = 1.0,
             output_root: str = "output", lr: float = 1e-4, wd: float = 1e-5,
             gpu: Optional[str] = None) -> None:
    """Delegate to umap.py to keep model-loading logic in a single place."""
    if gpu is not None:
        import os as _os
        _os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # Imported lazily so a missing umap-learn / MII install only breaks --umap.
    from umap import default_device, render_all
    device = default_device()
    out_dir = fig_root / "umap"
    render_all(
        dataset=dataset, k=k, ar=ar, out_dir=out_dir, device=device,
        lr=lr, wd=wd, output_root=output_root,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_dataset(s: str) -> str:
    """Accept 'hip' / 'covid' shorthand or the canonical eval_output names."""
    s = s.strip()
    aliases = {
        "hip": "hip734_cropped",
        "covid": "Covid",
    }
    return aliases.get(s.lower(), s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate every paper figure from eval_output/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_results.py --all
  python generate_results.py --dataset hip --preset mii_l1
  python generate_results.py --dataset covid --pseudolabels
  python generate_results.py --dataset covid --calibration-grid --dca-grid
  python generate_results.py --umap
""",
    )
    p.add_argument("--dataset", type=parse_dataset, default=None,
                   help="Dataset (hip / covid or canonical name)")
    p.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                   help="Named comparison group")
    p.add_argument("--all", action="store_true",
                   help="Render every preset + pseudolabels + grids + UMAP")
    p.add_argument("--all-presets", action="store_true",
                   help="Render every preset for the given --dataset")
    p.add_argument("--pseudolabels", action="store_true",
                   help="Render figures/{dataset}/pseudolabels/top1_pslabels.png")
    p.add_argument("--calibration-grid", action="store_true",
                   help="Render figures/covid/calibration_mii_vs_baseline_grid.png")
    p.add_argument("--dca-grid", action="store_true",
                   help="Render figures/covid/dca_mii_vs_baseline_grid.png")
    p.add_argument("--umap", action="store_true",
                   help="Render figures/umap/*.png (delegates to umap.py)")
    p.add_argument("--eval-root", default="eval_output",
                   help="Root for evaluation outputs (default: eval_output)")
    p.add_argument("--fig-root", default="figures",
                   help="Output root for figures (default: figures)")
    p.add_argument("--metrics", nargs="+", default=["top1", "auroc"],
                   choices=list(METRIC_CFG.keys()),
                   help="Metrics to plot in scatter figures")
    p.add_argument("--ymin", type=float, default=None,
                   help="Override y-axis minimum")
    p.add_argument("--ymax", type=float, default=None,
                   help="Override y-axis maximum")
    p.add_argument("--width", type=int, default=1000)
    p.add_argument("--height", type=int, default=700)

    p.add_argument("--umap-dataset", default="hip",
                   help="Dataset slug for --umap (default: hip)")
    p.add_argument("--umap-k", type=int, default=0)
    p.add_argument("--umap-ar", type=float, default=1.0)
    p.add_argument("--umap-output-root", default="output")
    p.add_argument("--umap-lr", type=float, default=1e-4,
                   help="LR used during ViT-MAE pretraining (default: 1e-4)")
    p.add_argument("--umap-wd", type=float, default=1e-5,
                   help="WD used during ViT-MAE pretraining (default: 1e-5)")
    p.add_argument("--gpu", default=None, type=str,
                   help="CUDA_VISIBLE_DEVICES for --umap")
    return p


def _yrange(args) -> Optional[tuple[float, float]]:
    if args.ymin is not None and args.ymax is not None:
        return (args.ymin, args.ymax)
    return None


def main() -> None:
    args = build_parser().parse_args()
    fig_root = Path(args.fig_root)

    if args.all:
        for ds in DATASETS:
            for preset_name, exps in PRESETS.items():
                run_preset(args.eval_root, ds, preset_name, exps, fig_root,
                           metrics=args.metrics, yrange=_yrange(args),
                           width=args.width, height=args.height)
            run_pseudolabels(args.eval_root, ds, fig_root,
                             width=args.width, height=args.height)
        # Calibration / DCA grids are paper-specific to Covid.
        run_calibration_grid(args.eval_root, "Covid", fig_root)
        run_dca_grid(args.eval_root, "Covid", fig_root)
        run_umap(fig_root, dataset=args.umap_dataset, k=args.umap_k,
                 ar=args.umap_ar, output_root=args.umap_output_root,
                 lr=args.umap_lr, wd=args.umap_wd, gpu=args.gpu)
        return

    if args.umap and not (args.preset or args.pseudolabels or args.calibration_grid
                          or args.dca_grid or args.all_presets):
        run_umap(fig_root, dataset=args.umap_dataset, k=args.umap_k,
                 ar=args.umap_ar, output_root=args.umap_output_root,
                 lr=args.umap_lr, wd=args.umap_wd, gpu=args.gpu)
        return

    if args.dataset is None:
        raise SystemExit("--dataset required (use --all to do every dataset)")

    if args.all_presets:
        for preset_name, exps in PRESETS.items():
            run_preset(args.eval_root, args.dataset, preset_name, exps,
                       fig_root, metrics=args.metrics, yrange=_yrange(args),
                       width=args.width, height=args.height)

    if args.preset:
        run_preset(args.eval_root, args.dataset, args.preset,
                   PRESETS[args.preset], fig_root, metrics=args.metrics,
                   yrange=_yrange(args), width=args.width, height=args.height)

    if args.pseudolabels:
        run_pseudolabels(args.eval_root, args.dataset, fig_root,
                         width=args.width, height=args.height)

    if args.calibration_grid:
        run_calibration_grid(args.eval_root, args.dataset, fig_root)

    if args.dca_grid:
        run_dca_grid(args.eval_root, args.dataset, fig_root)

    if args.umap:
        run_umap(fig_root, dataset=args.umap_dataset, k=args.umap_k,
                 ar=args.umap_ar, output_root=args.umap_output_root,
                 lr=args.umap_lr, wd=args.umap_wd, gpu=args.gpu)


if __name__ == "__main__":
    main()
