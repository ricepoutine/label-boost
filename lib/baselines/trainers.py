"""Per-method training epochs for SSL / KD baselines."""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.optim import Optimizer

from lib.baselines.augment import StrongAugment
from lib.baselines.ema import EMATeacher
from lib.baselines.losses import (
    fixmatch_unsupervised_loss,
    hard_ce_loss,
    kd_loss,
    mt_consistency_loss,
    one_hot,
    soft_ce_loss,
)
from lib.criterion import eps as CLS_EPS


def _interleave_loaders(dloader_l, dloader_u):
    """Yield (l_batch, u_batch) pairs; stop when the shorter loader exhausts."""
    if len(dloader_l.dataset) >= len(dloader_u.dataset):
        outer, inner, outer_labeled = dloader_l, dloader_u, True
    else:
        outer, inner, outer_labeled = dloader_u, dloader_l, False
    inner_iter = iter(inner)
    for outer_batch in outer:
        try:
            inner_batch = next(inner_iter)
        except StopIteration:
            break
        if outer_labeled:
            yield outer_batch, inner_batch
        else:
            yield inner_batch, outer_batch


def train_epoch_fixmatch(
    model: nn.Module,
    optimizer: Optimizer,
    dloader_l,
    dloader_u,
    lam_u: float,
    tau: float,
    pl_soft: bool,
    num_classes: int,
    device: torch.device,
    strong_aug: Optional[StrongAugment] = None,
) -> dict:
    """FixMatch: hard CE on L; confidence-gated PL from weak→strong on U."""
    model.train()
    strong_aug = strong_aug or StrongAugment()
    ce_total = 0.0
    n = 0
    mask_total = 0.0
    mask_n = 0
    u_total = 0.0
    u_n = 0

    if dloader_u is None:
        for img, lbl, _ in dloader_l:
            optimizer.zero_grad()
            loss, ce = hard_ce_loss(model, img, lbl, num_classes, device)
            loss.backward()
            optimizer.step()
            ce_total += ce
            n += 1
    else:
        for l_batch, u_batch in _interleave_loaders(dloader_l, dloader_u):
            optimizer.zero_grad()
            loss_l, ce = hard_ce_loss(model, l_batch[0], l_batch[1], num_classes, device)

            weak = u_batch[0].float().to(device)
            strong = strong_aug(weak)
            loss_u, mask_rate, u_scalar = fixmatch_unsupervised_loss(
                model, weak, strong, tau, pl_soft, num_classes, device,
            )
            (loss_l + lam_u * loss_u).backward()
            optimizer.step()

            ce_total += ce
            n += 1
            mask_total += mask_rate
            mask_n += 1
            if u_scalar > 0:
                u_total += u_scalar
                u_n += 1

    return {
        "hard_ce": ce_total / max(n, 1),
        "mask_rate": mask_total / mask_n if mask_n else None,
        "unsup_loss": u_total / u_n if u_n else None,
    }


def train_epoch_mean_teacher(
    model: nn.Module,
    ema: EMATeacher,
    optimizer: Optimizer,
    dloader_l,
    dloader_u,
    lam_u: float,
    num_classes: int,
    device: torch.device,
    strong_aug: Optional[StrongAugment] = None,
) -> dict:
    """Mean Teacher: hard CE on L; MSE consistency student(strong) vs EMA(weak) on U."""
    model.train()
    ema.eval()
    strong_aug = strong_aug or StrongAugment()
    ce_total = 0.0
    n = 0
    cons_total = 0.0
    cons_n = 0

    if dloader_u is None:
        for img, lbl, _ in dloader_l:
            optimizer.zero_grad()
            loss, ce = hard_ce_loss(model, img, lbl, num_classes, device)
            loss.backward()
            optimizer.step()
            ema.update(model)
            ce_total += ce
            n += 1
    else:
        for l_batch, u_batch in _interleave_loaders(dloader_l, dloader_u):
            optimizer.zero_grad()
            loss_l, ce = hard_ce_loss(model, l_batch[0], l_batch[1], num_classes, device)

            weak = u_batch[0].float().to(device)
            strong = strong_aug(weak)
            student_probs, _ = model(strong)
            with torch.no_grad():
                teacher_probs, _ = ema.teacher(weak)
            loss_cons = mt_consistency_loss(student_probs, teacher_probs)

            (loss_l + lam_u * loss_cons).backward()
            optimizer.step()
            ema.update(model)

            ce_total += ce
            n += 1
            cons_total += float(loss_cons.item())
            cons_n += 1

    return {
        "hard_ce": ce_total / max(n, 1),
        "consistency": cons_total / cons_n if cons_n else None,
    }


def train_epoch_kd(
    model: nn.Module,
    teacher: nn.Module,
    optimizer: Optimizer,
    dloader_l,
    dloader_u,
    lam: float,
    temperature: float,
    num_classes: int,
    device: torch.device,
    two_step: bool = False,
) -> dict:
    """Temperature KD on L (+ U when available). Teacher is frozen.

    ``two_step=False`` is the Hinton objective: one backward of
    hard CE + λ KD, then one optimizer step.

    ``two_step=True`` steps on the human-label loss first. The KD term is
    then forwarded again, so it sees the updated weights, and stepped
    separately. Unlabeled batches stay a single KD step in both modes.
    """
    model.train()
    teacher.eval()
    ce_total = 0.0
    n = 0
    kd_total = 0.0
    kd_n = 0

    def _kd_on_batch(img: torch.Tensor) -> torch.Tensor:
        image = img.float().to(device)
        logits, _, _ = model(image, return_logits=True)
        with torch.no_grad():
            t_probs, _ = teacher(image)
        return kd_loss(logits, t_probs, temperature)

    def _update_labeled(img: torch.Tensor, lbl: torch.Tensor) -> tuple[float, Optional[float]]:
        optimizer.zero_grad()
        loss_ce, ce = hard_ce_loss(model, img, lbl, num_classes, device)
        kd_scalar = None
        if two_step:
            loss_ce.backward()
            optimizer.step()
            if lam > 0:
                optimizer.zero_grad()
                loss_kd = _kd_on_batch(img)
                (lam * loss_kd).backward()
                optimizer.step()
                kd_scalar = float(loss_kd.item())
        elif lam > 0:
            loss_kd = _kd_on_batch(img)
            (loss_ce + lam * loss_kd).backward()
            optimizer.step()
            kd_scalar = float(loss_kd.item())
        else:
            loss_ce.backward()
            optimizer.step()
        return ce, kd_scalar

    if dloader_u is None:
        for img, lbl, _ in dloader_l:
            ce, kd_scalar = _update_labeled(img, lbl)
            if kd_scalar is not None:
                kd_total += kd_scalar
                kd_n += 1
            ce_total += ce
            n += 1
    else:
        for l_batch, u_batch in _interleave_loaders(dloader_l, dloader_u):
            ce, kd_l = _update_labeled(l_batch[0], l_batch[1])

            # Unlabeled: KD only
            if lam > 0:
                optimizer.zero_grad()
                loss_kd_u = _kd_on_batch(u_batch[0])
                (lam * loss_kd_u).backward()
                optimizer.step()
                kd_total += float(kd_l) + float(loss_kd_u.item())
                kd_n += 2

            ce_total += ce
            n += 1

    return {
        "hard_ce": ce_total / max(n, 1),
        "kd_loss": kd_total / kd_n if kd_n else None,
    }


def _student_forward_with_params(
    student: nn.Module,
    x: torch.Tensor,
    params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Forward student with substituted parameters (soft MPL one-step unroll)."""
    try:
        from torch.func import functional_call
    except ImportError:  # pragma: no cover
        from torch.nn.utils.stateless import functional_call  # type: ignore

    state = {**dict(student.named_buffers()), **params}
    out = functional_call(
        student, state, args=(x,), kwargs={"return_logits": True},
    )
    _, probs, _ = out
    return probs


def train_epoch_mpl(
    student: nn.Module,
    teacher: nn.Module,
    opt_s: Optimizer,
    opt_t: Optimizer,
    dloader_l,
    dloader_u,
    num_classes: int,
    device: torch.device,
    student_lr: float,
) -> dict:
    """Meta Pseudo Labels (soft, lean): co-trained ResNet18 teacher + student.

    Student learns from teacher soft PL on unlabeled data. Teacher receives
    supervised CE on labeled data plus a one-step soft meta-feedback signal.
    UDA teacher auxiliary loss is omitted (controlled simplification).
    """
    if dloader_u is None:
        # Fully labeled: supervised training for both networks
        return train_epoch_mpl_supervised_only(
            student, teacher, opt_s, opt_t, dloader_l, num_classes, device,
        )

    student.train()
    teacher.train()
    ce_total = 0.0
    n = 0
    u_total = 0.0
    u_n = 0
    meta_total = 0.0
    meta_n = 0

    for l_batch, u_batch in _interleave_loaders(dloader_l, dloader_u):
        xl = l_batch[0].float().to(device)
        yl = l_batch[1].to(device)
        xu = u_batch[0].float().to(device)
        yl_oh = one_hot(yl, num_classes, device)

        # --- Student update on unlabeled soft PL (detached) ---
        with torch.no_grad():
            pl_det, _ = teacher(xu)
        opt_s.zero_grad()
        pred_u, _ = student(xu)
        loss_u = soft_ce_loss(pred_u, pl_det)
        loss_u.backward()
        opt_s.step()

        # --- Teacher: supervised aux + soft one-step meta feedback ---
        opt_t.zero_grad()
        pred_t_l, _ = teacher(xl)
        loss_t_sup = soft_ce_loss(pred_t_l, yl_oh)

        # Soft PL with grad through teacher; one-step student unroll
        pl_soft, _ = teacher(xu)
        pred_u_meta, _ = student(xu)
        # Connect soft_ce to teacher via pl_soft; student side needs create_graph
        log_p = torch.log(pred_u_meta.clamp(min=CLS_EPS))
        loss_u_meta = -torch.mean(torch.sum(log_p * pl_soft, dim=1))

        s_named = list(student.named_parameters())
        s_params = [p for _, p in s_named]
        grads = torch.autograd.grad(
            loss_u_meta,
            s_params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )

        fast_params = {}
        for (name, p), g in zip(s_named, grads):
            if g is None:
                fast_params[name] = p
            else:
                fast_params[name] = p - student_lr * g

        pred_l_fast = _student_forward_with_params(student, xl, fast_params)
        loss_meta = soft_ce_loss(pred_l_fast, yl_oh)

        (loss_t_sup + loss_meta).backward()
        opt_t.step()

        # Diagnostic: student CE on labeled (no grad)
        with torch.no_grad():
            pred_l, _ = student(xl)
            ce_total += float(soft_ce_loss(pred_l, yl_oh).item())
        n += 1
        u_total += float(loss_u.item())
        u_n += 1
        meta_total += float(loss_meta.item())
        meta_n += 1

    return {
        "hard_ce": ce_total / max(n, 1),
        "unsup_loss": u_total / u_n if u_n else None,
        "meta_loss": meta_total / meta_n if meta_n else None,
    }


def train_epoch_mpl_supervised_only(
    student: nn.Module,
    teacher: nn.Module,
    opt_s: Optimizer,
    opt_t: Optimizer,
    dloader_l,
    num_classes: int,
    device: torch.device,
) -> dict:
    """When ar=1.0 there is no unlabeled pool; train both on hard labels."""
    student.train()
    teacher.train()
    ce_total = 0.0
    n = 0
    for img, lbl, _ in dloader_l:
        opt_s.zero_grad()
        loss_s, ce = hard_ce_loss(student, img, lbl, num_classes, device)
        loss_s.backward()
        opt_s.step()

        opt_t.zero_grad()
        loss_t, _ = hard_ce_loss(teacher, img, lbl, num_classes, device)
        loss_t.backward()
        opt_t.step()

        ce_total += ce
        n += 1
    return {"hard_ce": ce_total / max(n, 1), "unsup_loss": None, "meta_loss": None}


def train_epoch_supervised_finetune(
    model: nn.Module,
    optimizer: Optimizer,
    dloader_l,
    num_classes: int,
    device: torch.device,
) -> dict:
    """Labeled-only finetune (MPL post-phase / ar=1 fallback)."""
    model.train()
    ce_total = 0.0
    n = 0
    for img, lbl, _ in dloader_l:
        optimizer.zero_grad()
        loss, ce = hard_ce_loss(model, img, lbl, num_classes, device)
        loss.backward()
        optimizer.step()
        ce_total += ce
        n += 1
    return {"hard_ce": ce_total / max(n, 1)}
