"""Loss helpers for FixMatch, Mean Teacher, Meta Pseudo Labels, and temperature KD."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from lib.criterion import eps as CLS_EPS


def one_hot(labels: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    oh = torch.zeros(labels.size(0), num_classes, device=device)
    oh.scatter_(1, labels.long().view(-1, 1), 1.0)
    return oh


def soft_ce_loss(probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean CE between student probabilities and a soft/hard target distribution."""
    log_p = torch.log(probs.clamp(min=CLS_EPS))
    return -torch.mean(torch.sum(log_p * target, dim=1))


def hard_ce_loss(
    model: torch.nn.Module,
    img: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Supervised CE on hard labels. Returns (loss, scalar)."""
    image = img.float().to(device)
    target = one_hot(labels.to(device), num_classes, device)
    pred, _ = model(image)
    loss = soft_ce_loss(pred, target)
    return loss, float(loss.item())


def fixmatch_unsupervised_loss(
    model: torch.nn.Module,
    weak: torch.Tensor,
    strong: torch.Tensor,
    tau: float,
    pl_soft: bool,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, float, float]:
    """FixMatch U-loss: PL from weak view, CE on strong view, confidence gate.

    Returns (loss, mask_rate, scalar_loss). Loss is 0 if no samples pass the gate.
    """
    weak = weak.float().to(device)
    strong = strong.float().to(device)

    with torch.no_grad():
        p_weak, _ = model(weak)
        conf, pseudo_idx = torch.max(p_weak, dim=1)
        mask = conf.ge(tau).float()

    if mask.sum() < 1:
        zero = strong.new_zeros(())
        return zero, 0.0, 0.0

    if pl_soft:
        target = p_weak
    else:
        target = one_hot(pseudo_idx, num_classes, device)

    p_strong, _ = model(strong)
    log_p = torch.log(p_strong.clamp(min=CLS_EPS))
    per_ex = -torch.sum(log_p * target, dim=1)
    # Average over confident examples only (FixMatch convention)
    loss = (per_ex * mask).sum() / mask.sum().clamp(min=1.0)
    return loss, float(mask.mean().item()), float(loss.item())


def mt_consistency_loss(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
) -> torch.Tensor:
    """Mean Teacher consistency: MSE between student and EMA-teacher softmax."""
    return F.mse_loss(student_probs, teacher_probs)


def kd_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Hinton KD: T^2 * KL(softmax(z_s/T) || softmax(z_t/T)).

    Teacher probs are converted to logits via log (equivalent under temperature
    softmax up to a per-sample constant).
    """
    t = temperature
    teacher_logits = torch.log(teacher_probs.clamp(min=CLS_EPS))
    log_p_s = F.log_softmax(student_logits / t, dim=1)
    p_t = F.softmax(teacher_logits / t, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (t * t)
