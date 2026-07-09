"""
Saliency-specific loss functions and evaluation metrics.

Loss:
    Combined = w_kl * KLDiv + w_cc * (1 - CC) + w_mse * MSE

Metrics (all standard in saliency benchmarks):
    KL  – KL divergence  (lower is better)
    CC  – Pearson correlation coefficient  (higher is better)
    NSS – Normalised scanpath saliency     (higher is better)
"""

import torch
import torch.nn.functional as F

_EPS = 1e-8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise(x: torch.Tensor) -> torch.Tensor:
    """Per-sample min-max normalise to [0, 1], shape (B, 1, H, W)."""
    b = x.shape[0]
    flat = x.view(b, -1)
    mn = flat.min(dim=1).values.view(b, 1, 1, 1)
    mx = flat.max(dim=1).values.view(b, 1, 1, 1)
    return (x - mn) / (mx - mn + _EPS)


def _to_prob(x: torch.Tensor) -> torch.Tensor:
    """Normalise so the spatial sum = 1 (treat map as a probability distribution)."""
    b = x.shape[0]
    flat = x.view(b, -1)
    s = flat.sum(dim=1, keepdim=True).view(b, 1, 1, 1)
    return x / (s + _EPS)


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def kl_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """KL divergence: KL(target || pred), both normalised to probability distributions."""
    p = _to_prob(target)
    q = _to_prob(pred)
    return (p * torch.log(p / (q + _EPS) + _EPS)).sum(dim=[1, 2, 3]).mean()


def cc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Pearson CC so it can be minimised."""
    b = pred.shape[0]
    p = pred.view(b, -1)
    t = target.view(b, -1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    num = (p * t).sum(dim=1)
    denom = (p.norm(dim=1) * t.norm(dim=1)).clamp(min=_EPS)
    return (1.0 - (num / denom)).mean()


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


class SaliencyLoss(torch.nn.Module):
    def __init__(self, w_kl: float = 1.0, w_cc: float = 1.0, w_mse: float = 0.5):
        super().__init__()
        self.w_kl = w_kl
        self.w_cc = w_cc
        self.w_mse = w_mse

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, float]]:
        l_kl = kl_loss(pred, target)
        l_cc = cc_loss(pred, target)
        l_mse = mse_loss(pred, target)
        total = self.w_kl * l_kl + self.w_cc * l_cc + self.w_mse * l_mse
        breakdown = {
            "loss/kl": l_kl.item(),
            "loss/cc": l_cc.item(),
            "loss/mse": l_mse.item(),
            "loss/total": total.item(),
        }
        return total, breakdown


# ---------------------------------------------------------------------------
# Metrics  (no_grad context assumed at call site)
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Compute KL, CC and NSS for a batch. Returns Python floats."""
    b = pred.shape[0]

    # KL
    kl = kl_loss(pred, target).item()

    # CC
    p = pred.view(b, -1)
    t = target.view(b, -1)
    p_c = p - p.mean(dim=1, keepdim=True)
    t_c = t - t.mean(dim=1, keepdim=True)
    num = (p_c * t_c).sum(dim=1)
    denom = (p_c.norm(dim=1) * t_c.norm(dim=1)).clamp(min=_EPS)
    cc = (num / denom).mean().item()

    # NSS – evaluate at fixation locations (pixels > 0.5 after normalising target)
    target_norm = _normalise(target)
    fix_mask = (target_norm > 0.5).float()
    pred_z = pred.view(b, -1)
    pred_z = (pred_z - pred_z.mean(dim=1, keepdim=True)) / (pred_z.std(dim=1, keepdim=True) + _EPS)
    pred_z = pred_z.view_as(pred)
    nss_per = (pred_z * fix_mask).sum(dim=[1, 2, 3]) / (fix_mask.sum(dim=[1, 2, 3]) + _EPS)
    nss = nss_per.mean().item()

    return {"metric/kl": kl, "metric/cc": cc, "metric/nss": nss}
