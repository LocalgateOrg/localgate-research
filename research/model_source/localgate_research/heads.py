"""Losses and probability decoders for five-trial success counts."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import HEAD_NAMES

TRIALS = 5
_WIDTHS = dict(zip(HEAD_NAMES, (6, 5, 1, 1, 2, 7, 6), strict=True))


def output_width(head: str) -> int:
    try:
        return _WIDTHS[head]
    except KeyError as exc:
        raise ValueError(f"Unknown head: {head!r}") from exc


def _validate_logits(logits: Tensor, head: str) -> None:
    if logits.ndim != 2 or logits.shape[1] != output_width(head) or not logits.shape[0]:
        raise ValueError(f"Expected nonempty logits with shape (n, {output_width(head)})")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("Logits must be finite floating-point values")


def _validate_counts(counts: Tensor, logits: Tensor) -> Tensor:
    if counts.ndim != 1 or counts.shape[0] != logits.shape[0]:
        raise ValueError("Counts must be a vector aligned with the logits")
    values = counts.to(device=logits.device, dtype=torch.float64)
    if not torch.isfinite(values).all() or not ((values >= 0) & (values <= TRIALS)).all():
        raise ValueError("Counts must be finite values in 0..5")
    if not (values == values.floor()).all():
        raise ValueError("Counts must be integers")
    return values


def _count_support(logits: Tensor) -> tuple[Tensor, Tensor]:
    support = torch.arange(TRIALS + 1, device=logits.device, dtype=torch.float64)
    log_choose = torch.tensor(
        [math.log(math.comb(TRIALS, count)) for count in range(TRIALS + 1)],
        device=logits.device,
        dtype=torch.float64,
    )
    return support, log_choose


def _binomial_log_pmf(logit: Tensor) -> Tensor:
    support, log_choose = _count_support(logit)
    return (
        log_choose
        + support * F.logsigmoid(logit[:, None])
        + (TRIALS - support) * F.logsigmoid(-logit[:, None])
    )


def _beta_parameters(logits: Tensor) -> tuple[Tensor, Tensor]:
    parameters = F.softplus(logits) + 1e-4
    return parameters[:, :1], parameters[:, 1:2]


def _beta_binomial_log_pmf(logits: Tensor) -> Tensor:
    support, log_choose = _count_support(logits)
    alpha, beta = _beta_parameters(logits)
    return (
        log_choose
        + torch.lgamma(support + alpha)
        + torch.lgamma(TRIALS - support + beta)
        - torch.lgamma(TRIALS + alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + torch.lgamma(alpha + beta)
    )


def loss(logits: Tensor, counts: Tensor, head: str) -> Tensor:
    """Return the selected head's scalar training loss."""
    _validate_logits(logits, head)
    counts = _validate_counts(counts, logits)
    # Keep count-likelihood arithmetic in float64 even when the encoder uses BF16.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        values = logits.to(torch.float64)
        if head == "softmax":
            return F.cross_entropy(values, counts.long())
        if head == "corn":
            thresholds = torch.arange(TRIALS, device=values.device)
            eligible = counts[:, None] >= thresholds
            targets = (counts[:, None] > thresholds).to(values.dtype)
            tasks = F.binary_cross_entropy_with_logits(values, targets, reduction="none")
            return tasks[eligible].sum() / eligible.sum()
        if head == "regression":
            return F.mse_loss(values[:, 0].sigmoid(), counts / TRIALS)
        if head == "binomial":
            log_pmf = _binomial_log_pmf(values[:, 0])
            return -log_pmf.gather(1, counts.long()[:, None]).mean()
        if head == "beta_binomial":
            log_pmf = _beta_binomial_log_pmf(values)
            return -log_pmf.gather(1, counts.long()[:, None]).mean()
        if head == "twin":
            softmax_log_pmf = F.log_softmax(values[:, :6], dim=-1)
            binomial_log_pmf = _binomial_log_pmf(values[:, 6])
            ce = F.nll_loss(softmax_log_pmf, counts.long())
            count_nll = -binomial_log_pmf.gather(1, counts.long()[:, None]).mean()
            consistency = (
                (binomial_log_pmf.exp() * (binomial_log_pmf - softmax_log_pmf)).sum(dim=-1).mean()
            )
            return ce + count_nll + 0.5 * consistency
        cumulative = values.softmax(dim=-1).cumsum(dim=-1)[:, :-1]
        thresholds = torch.arange(TRIALS, device=values.device)
        observed = (counts[:, None] <= thresholds).to(values.dtype)
        return (cumulative - observed).square().mean()


def decode(logits: Tensor, head: str) -> tuple[Tensor, Tensor | None]:
    """Return the mean success probability and the native count PMF, when defined."""
    _validate_logits(logits, head)
    with torch.autocast(device_type=logits.device.type, enabled=False):
        values = logits.to(torch.float64)
        if head == "regression":
            return values[:, 0].sigmoid(), None
        if head == "corn":
            survival = values.sigmoid().cumprod(dim=-1)
            pmf = torch.cat(
                (1 - survival[:, :1], survival[:, :-1] - survival[:, 1:], survival[:, -1:]),
                dim=-1,
            )
            return survival.sum(dim=-1) / TRIALS, pmf
        elif head == "binomial":
            pmf = _binomial_log_pmf(values[:, 0]).exp()
            return values[:, 0].sigmoid(), pmf
        elif head == "beta_binomial":
            pmf = _beta_binomial_log_pmf(values).exp()
            alpha, beta = _beta_parameters(values)
            return (alpha / (alpha + beta))[:, 0], pmf
        else:
            pmf = values[:, :6].softmax(dim=-1)
        support, _ = _count_support(values)
        return (pmf * support).sum(dim=-1) / TRIALS, pmf
