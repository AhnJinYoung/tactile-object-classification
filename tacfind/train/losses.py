"""Loss functions of the training stages (pure functions).

contrastive_loss: touch embedding vs. object library (stage 1); confidence_loss: matcher confidence (stage 2);
chamfer_loss: auxiliary supervision of the estimated contact points of `matcher=latent_head`;
uniformity_loss: spreads the library embeddings on the sphere (counteracts their collapse in stage 1).
Rows without a label are skipped; when none is valid every loss returns a zero that is still attached to the graph.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _masked_mean(per_row: Tensor, valid: Tensor) -> Tensor:
    """Mean of per_row (B,) over valid rows; a graph-attached zero when none is valid."""
    return torch.where(valid, per_row, 0.0).sum() / valid.sum().clamp(min=1)


def contrastive_loss(e_touch: Tensor, z_lib: Tensor, touched: Tensor, temperature: float) -> tuple[Tensor, Tensor]:
    """InfoNCE of the touch embedding against the whole object library, with cosine similarity.

    Args:
        e_touch: (B, dim) touch embeddings.
        z_lib: (n_lib, dim) object embeddings (TargetCode.z of the library).
        touched: (B,) long, library index of the touched object; -1 = nothing touched (row skipped).
        temperature: softmax temperature.
    Returns:
        loss (scalar), top-1 accuracy over valid rows (scalar, detached; NaN when no row is valid).
    """
    valid = touched >= 0
    logits = F.normalize(e_touch, dim=-1) @ F.normalize(z_lib, dim=-1).T / temperature  # (B, n_lib)
    labels = touched.clamp(min=0)
    loss = _masked_mean(F.cross_entropy(logits, labels, reduction="none"), valid)
    acc = ((logits.argmax(-1) == labels) & valid).sum().detach() / valid.sum()  # 0 / 0 = NaN; no host sync
    return loss, acc


def confidence_loss(conf_logit: Tensor, conf_gt: Tensor) -> Tensor:
    """BCE-with-logits against a label in [0, 1] (soft contact ratio or binary flag).

    Args:
        conf_logit: (B,) logits. conf_gt: (B,) labels in [0, 1].
    """
    bce = F.binary_cross_entropy_with_logits(conf_logit, conf_gt.to(conf_logit.dtype), reduction="none")
    return _masked_mean(bce, torch.ones_like(bce, dtype=torch.bool))


def chamfer_loss(est_points: Tensor, touch_xyz: Tensor, touch_mask: Tensor, pos_scale: float = 0.1) -> Tensor:
    """Symmetric masked Chamfer distance (mean squared distance, in units of pos_scale^2) between centred clouds.

    Each cloud is centred on its own (masked) centroid first: only relative geometry is supervised, because the
    estimated points carry no absolute position.

    Args:
        est_points: (B, K, 3) estimated contact points (m).
        touch_xyz: (B, L, 3) real contact positions (m). touch_mask: (B, L) bool, True = real contact.
        pos_scale: length unit (m) of the distances; the matcher's `pos_scale`, so the loss is O(1).
    Returns:
        scalar loss over rows with at least one real contact.
    """
    m = touch_mask[..., None].to(touch_xyz.dtype)
    a = (est_points - est_points.mean(1, keepdim=True)) / pos_scale
    b = (touch_xyz - (touch_xyz * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)) / pos_scale
    d = (a[:, :, None] - b[:, None]).square().sum(-1)  # (B, K, L); no sqrt: its gradient at 0 is NaN
    est_to_touch = d.masked_fill(~touch_mask[:, None], 1e9).min(2).values.mean(1)  # finite fill keeps grads finite
    touch_to_est = (d.min(1).values * m[..., 0]).sum(1) / m[..., 0].sum(1).clamp(min=1)
    return _masked_mean(est_to_touch + touch_to_est, touch_mask.any(1))


def uniformity_loss(z: Tensor, t: float = 2.0) -> Tensor:
    """Uniformity of L2-normalised embeddings (Wang & Isola 2020): log mean_{i != j} exp(-t * |z_i - z_j|^2).

    Args:
        z: (n, dim) embeddings, e.g. the object library's TargetCode.z.
        t: kernel sharpness.
    Returns:
        scalar in [-4t, 0]; lower = more uniform. A graph-attached zero when n < 2.
    """
    n = z.shape[0]
    if n < 2:
        return z.sum() * 0.0
    z = F.normalize(z, dim=-1)
    sq_dist = (2 - 2 * z @ z.T).clamp(min=0)  # |a - b|^2 for unit vectors; smooth, unlike cdist at 0
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    return torch.logsumexp((-t * sq_dist).masked_fill(eye, float("-inf")).flatten(), 0) - math.log(n * (n - 1))
