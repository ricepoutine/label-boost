# MediVision — Label-Boost Knowledge Distillation for Medical Imaging

Reference implementation for the NeurIPS submission. The repository covers
unsupervised feature pretraining of [facebook/vit-mae-large](https://huggingface.co/facebook/vit-mae-large),
linear-probe training for ViT-MAE / [MedImageInsights (MII)](https://huggingface.co/microsoft/medimageinsightmodel)
teachers, label-boost knowledge distillation into a ResNet18 student, and full
publication-grade evaluation (calibration, decision-curve analysis, paired
t-tests, scatter plots) for hip dysplasia (`hip734_cropped`) and COVID-19
chest X-ray (`Covid`) datasets.

## Top-level Scripts

After cleanup, the project consists of exactly **five** top-level scripts:

| Script | Purpose |
| --- | --- |
| [`finetune.py`](finetune.py) | ViT-MAE unsupervised reconstruction pretraining → encoder checkpoints under `output/{dataset}/lr{lr}_wd{wd}/ViTMAE/.../checkpoints/best.pth.tar`. |
| [`probe.py`](probe.py) | Train the linear probe on top of a ViT-MAE / MII encoder. Auto-invoked by `evaluate.py`; can also run standalone. |
| [`evaluate.py`](evaluate.py) | Train every student configuration (`ResNet18`, `ResNet18_soft`, `ResNet18_{ViTMAE,MII}_l{0.5,1.0,1.5}` ± `_ablation`) over five-fold CV and write per-fold predictions, summaries, and PHAIR metrics under `eval_output/`. |
| [`umap.py`](umap.py) | UMAP feature projections (PNG): MII untrained, ViT-MAE pretrained, ViT-MAE finetuned. |
| [`generate_results.py`](generate_results.py) | Single entry point for every paper figure under `figures/{covid,hip}/*` and `figures/umap/*` and the calibration / DCA grids. |

Supporting libraries live under `lib/`, model wrappers under `models/`, the
shared evaluation suite under `phair_eval_suite/` (do **not** modify), and
patient-split caches under `PT_configs/`.

---

## Installation

```bash
# 1. Conda environment (the repo is pinned for torch 2.1.x).
conda create -n torch python=3.10 -y
conda activate torch

# 2. Install pinned dependencies.
pip install -r requirements.txt
```

**Always** run this project inside the `torch` conda env — it is the only
environment with the pinned `torch==2.1.1` / `transformers==4.38.2` / matching
`tokenizers` stack required by the saved checkpoints.

The first run will download `facebook/vit-mae-large` from Hugging Face. If
the cluster has no internet, set `HF_HOME` and pre-download with `huggingface-cli`.

GPU notes:

- CUDA is auto-detected (`evaluate.default_device`). Use `--gpu 0` (or
  `--gpu 0,1`) on every script that exposes the flag.
- Apple Silicon falls back to MPS where supported; some teachers (MII)
  expect CUDA, so prefer a CUDA box for full reproduction.

---

## Data Setup

Datasets live under `datasets/`. Each dataset is a flat
[`torchvision.datasets.ImageFolder`](https://pytorch.org/vision/stable/datasets.html#torchvision.datasets.ImageFolder)
with two class subdirectories — exactly:

```
datasets/
├── hip734_cropped/
│   ├── absent/      # normal hips
│   └── present/     # dysplastic hips
└── Covid/
    ├── absent/      # negative chest X-rays
    └── present/     # positive chest X-rays
```

Filenames must encode the patient ID so [`lib.dataloader.split_by_patient`](lib/dataloader.py)
can group images per subject during k-fold cross-validation. The cropping /
preprocessing pipeline that produced these layouts is dataset-specific (e.g.
`hip734_cropped` is generated upstream and not included here).

Patient-split + per-channel normalization caches are written to
`PT_configs/{dataset}/tr0.7_vr0.1_s0.json` on first run and reused for every
later run with the same `train_ratio / valid_ratio / seed`.

### MII weights

`models/MedImageInsights/` is **not** vendored in this repository. Drop the
released MedImageInsights weights into:

```
models/MedImageInsights/
├── 2024.09.27/
│   ├── medimageinsigt-v1.0.0.pt
│   └── language_model.pt
└── medimageinsightmodel.py   # MII inference wrapper from the upstream release
```

Refer to the official [MedImageInsights release](https://huggingface.co/microsoft/medimageinsightmodel)
for the exact files. Without these, only ViT-MAE-based experiments and
generate-results figures that don't touch MII will work.

---

## Reproducing the Paper End-to-End

Run these in order. Every command is idempotent — re-running picks up at the
next missing checkpoint / summary file.

### 1. Pretrain the ViT-MAE encoder (one run per dataset)

```bash
# 5-fold ViT-MAE pretraining. Default args (epochs 150, lr 1e-4, wd 1e-5)
# match the paper. Use --expanded-train-set to also use unlabeled data.
python finetune.py --dataset hip   --gpu 0
python finetune.py --dataset covid --gpu 0
```

Writes `output/{dataset}/lr1e-4_wd1e-5/ViTMAE/tr0.7_s0_k{k}/checkpoints/best.pth.tar`
for `k ∈ [0, 5)`. These are the encoder weights that `probe.py` and
`evaluate.load_teacher` consume.

### 2. Train every student configuration

```bash
# Trains ResNet18, ResNet18_soft, ResNet18_{ViTMAE,MII}_l{0.5,1.0,1.5} ± _ablation
# over annotation ratios 0.1 .. 1.0, five folds each.  Auto-runs probe.py for
# any teacher whose linear head is missing.
python evaluate.py --dataset hip   --gpu 0
python evaluate.py --dataset covid --gpu 0
```

To run a single experiment / AR (e.g. for debugging):

```bash
python evaluate.py --dataset hip --experiment ResNet18_MII_l1.0 --ar 1.0 --gpu 0
```

Writes per-fold predictions, summaries, and PHAIR metrics under
`eval_output/{dataset}/{experiment}/ar{X}/`.

### 3. Generate every paper figure

```bash
# Full sweep (all presets, both datasets, pseudolabel + grid + UMAP figures).
python generate_results.py --all
```

Produces:

- `figures/hip/{baseline,vitmae_l1,vitmae_lambdas,vitmae_ablation,mii_l1,mii_lambdas,mii_ablation,all_l1,all}/{accuracy,auroc}.png`
- `figures/covid/{baseline,vitmae_l1,vitmae_lambdas,vitmae_ablation,mii_l1,mii_lambdas,mii_ablation,all_l1,all}/{accuracy,auroc}.png`
- `figures/{hip,covid}/pseudolabels/top1_pslabels.png`
- `figures/covid/calibration_mii_vs_baseline_grid.png`
- `figures/covid/dca_mii_vs_baseline_grid.png`
- `figures/umap/{mii_untrained,vitmae_base,vitmae_finetuned}_umap.png`

---

## Per-Figure Reproduction

If you only need a subset, the following invocations map one-to-one to the
sub-folders in the paper repo (`thesis_paper/figures/...`).

| Figure (paper-repo path) | Command |
| --- | --- |
| `figures/hip/baseline/{accuracy,auroc}.png` | `python generate_results.py --dataset hip --preset baseline` |
| `figures/hip/mii_l1/{accuracy,auroc}.png` | `python generate_results.py --dataset hip --preset mii_l1` |
| `figures/hip/mii_lambdas/{accuracy,auroc}.png` | `python generate_results.py --dataset hip --preset mii_lambdas` |
| `figures/hip/mii_ablation/{accuracy,auroc}.png` | `python generate_results.py --dataset hip --preset mii_ablation` |
| `figures/hip/vitmae_l1/{accuracy,auroc}.png` | `python generate_results.py --dataset hip --preset vitmae_l1` |
| `figures/hip/vitmae_lambdas/{accuracy,auroc}.png` | `python generate_results.py --dataset hip --preset vitmae_lambdas` |
| `figures/hip/vitmae_ablation/{accuracy,auroc}.png` | `python generate_results.py --dataset hip --preset vitmae_ablation` |
| `figures/hip/all_l1/{accuracy,auroc}.png` | `python generate_results.py --dataset hip --preset all_l1` |
| `figures/hip/all/{accuracy,auroc}.png` | `python generate_results.py --dataset hip --preset all` |
| `figures/hip/pseudolabels/top1_pslabels.png` | `python generate_results.py --dataset hip --pseudolabels` |
| `figures/covid/*` (analogues of the above) | swap `--dataset hip` for `--dataset covid`. |
| `figures/covid/calibration_mii_vs_baseline_grid.png` | `python generate_results.py --dataset covid --calibration-grid` |
| `figures/covid/dca_mii_vs_baseline_grid.png` | `python generate_results.py --dataset covid --dca-grid` |
| `figures/umap/mii_untrained_umap.png` | `python umap.py --dataset hip --model mii --variant untrained` |
| `figures/umap/vitmae_base_umap.png` | `python umap.py --dataset hip --model vitmae --variant base` |
| `figures/umap/vitmae_finetuned_umap.png` | `python umap.py --dataset hip --model vitmae --variant finetuned` |

`generate_results.py --dataset {ds} --all-presets` runs every preset for one
dataset in a single shot. `python generate_results.py --umap` re-renders all
three UMAP PNGs.

Tables and significance markdowns (`tables.csv`, `tables.md`,
`significance.csv`, `significance.md`) are written next to the scatter PNGs.

---

## Output Directory Reference

```
output/                                              # ViT-MAE encoder checkpoints
└── {dataset}/lr{lr}_wd{wd}/ViTMAE/tr0.7_s0_k{k}/
    ├── checkpoints/best.pth.tar
    └── events.out.tfevents.*                        # MAE training TB

eval_output/                                         # Student-side results
└── {dataset}/{experiment}/
    ├── summary_all_ar.csv                           # cross-AR aggregates
    └── ar{ar}/
        ├── checkpoints/k{k}_best.pth.tar            # student model
        ├── checkpoints/linear_k{k}_best.pth.tar     # teacher head (probe.py)
        ├── fold_{k}_predictions.json                # per-fold y_true/y_proba
        ├── summary.csv                              # per-fold + mean/std rows
        ├── aggregate_metrics.json                   # PHAIR aggregate metrics
        ├── pooled_*.png                             # PHAIR pooled plots
        └── k{k}/                                    # per-fold TensorBoard

figures/                                             # Paper figures (this repo)
├── hip/<preset>/{accuracy,auroc,tables.*,significance.*}
├── covid/<preset>/...
├── covid/{calibration,dca}_mii_vs_baseline_grid.png
├── {hip,covid}/pseudolabels/top1_pslabels.png
└── umap/{mii_untrained,vitmae_base,vitmae_finetuned}_umap.png
```

---

## Caveats

- Legacy code paths from the older `main.py` / `main_transformer.py`
  (KNN-based pseudolabels, pulse / step alpha schedules, the `VegaCNN`
  classifier, and the soft-label JSON cache) were dropped during the
  cleanup. Only the experiments listed in `evaluate.EXPERIMENTS` and the
  presets in `generate_results.PRESETS` are reproducible from this release.
- `phair_eval_suite/` is vendored from the lab's shared repository and is
  imported by `evaluate.py` / `generate_results.py` without modification.
  The calibration / DCA grid logic in `generate_results.py` re-uses the
  same LOWESS-smoothing and threshold-sweep formulae as
  [`phair_eval_suite.core_eval_functions`](phair_eval_suite/core_eval_functions.py)
  but draws onto matplotlib subplot axes for the multi-AR grid.
- The script file is named `umap.py` per project convention. Internally
  it strips the project root from `sys.path` while resolving the
  `umap-learn` library so the local file does not shadow the third-party
  package.
