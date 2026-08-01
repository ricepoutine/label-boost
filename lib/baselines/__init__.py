"""SOTA SSL / KD baselines for fair comparison against Label-Boost."""

from lib.baselines.ema import EMATeacher
from lib.baselines.losses import (
    fixmatch_unsupervised_loss,
    hard_ce_loss,
    kd_loss,
    mt_consistency_loss,
    soft_ce_loss,
)
from lib.baselines.trainers import (
    train_epoch_fixmatch,
    train_epoch_kd,
    train_epoch_mean_teacher,
    train_epoch_mpl,
    train_epoch_supervised_finetune,
)

__all__ = [
    "EMATeacher",
    "fixmatch_unsupervised_loss",
    "hard_ce_loss",
    "kd_loss",
    "mt_consistency_loss",
    "soft_ce_loss",
    "train_epoch_fixmatch",
    "train_epoch_kd",
    "train_epoch_mean_teacher",
    "train_epoch_mpl",
    "train_epoch_supervised_finetune",
]
