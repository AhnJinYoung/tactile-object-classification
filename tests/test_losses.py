"""Tests of the training losses on hand-computable cases and on the no-valid-row case."""

import math

import torch

from tacfind.train.losses import chamfer_loss, confidence_loss, contrastive_loss, uniformity_loss

torch.set_num_threads(4)


def assert_attached_zero(loss: torch.Tensor, leaf: torch.Tensor):
    assert loss.item() == 0.0 and loss.requires_grad
    loss.backward()
    assert torch.isfinite(leaf.grad).all()


def test_contrastive_hand_case():
    # Cosine similarities are the identity -> logits [[1, 0], [0, 1]] at temperature 1; the -1 row is skipped.
    e_touch = torch.tensor([[1.0, 0.0], [0.0, 1.0], [5.0, -3.0]])
    z_lib = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    loss, acc = contrastive_loss(e_touch, z_lib, torch.tensor([0, 1, -1]), temperature=1.0)
    assert math.isclose(loss.item(), math.log(1 + math.exp(-1)), rel_tol=1e-6) and acc.item() == 1.0
    loss, acc = contrastive_loss(e_touch[:2], z_lib, torch.tensor([1, 1]), temperature=0.5)
    expected = (math.log(1 + math.exp(2)) + math.log(1 + math.exp(-2))) / 2
    assert math.isclose(loss.item(), expected, rel_tol=1e-6) and acc.item() == 0.5


def test_contrastive_no_valid_row():
    e_touch = torch.randn(3, 4, requires_grad=True)
    loss, acc = contrastive_loss(e_touch, torch.randn(5, 4), torch.full((3,), -1), temperature=0.1)
    assert torch.isnan(acc)
    assert_attached_zero(loss, e_touch)


def test_confidence_hand_case_and_empty():
    loss = confidence_loss(torch.tensor([0.0, 0.0]), torch.tensor([0.5, 1.0]))
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)
    loss = confidence_loss(torch.tensor([2.0]), torch.tensor([1]))  # integer flag label
    assert math.isclose(loss.item(), math.log(1 + math.exp(-2)), rel_tol=1e-6)
    logit = torch.zeros(0, requires_grad=True)
    assert_attached_zero(confidence_loss(logit, torch.zeros(0)), logit)


def test_chamfer_hand_case():
    # est centred: (-1,0,0), (1,0,0); one real touch point (centred: origin) + one padded point far away.
    est = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    touch = torch.tensor([[[5.0, 5.0, 5.0], [100.0, 0.0, 0.0]]])
    mask = torch.tensor([[True, False]])
    assert math.isclose(chamfer_loss(est, touch, mask, pos_scale=1.0).item(), 1.0 + 1.0, rel_tol=1e-6)
    # Default unit 0.1 m: the same geometry at a tenth of the size gives the same loss.
    assert math.isclose(chamfer_loss(0.1 * est, 0.1 * touch, mask).item(), 2.0, rel_tol=1e-5)


def test_chamfer_translation_and_invalid_rows():
    pts = torch.randn(2, 6, 3)
    est = (pts + torch.tensor([1.0, -2.0, 3.0])).requires_grad_()
    mask = torch.tensor([[True] * 6, [False] * 6])
    loss = chamfer_loss(est, pts, mask)  # row 0: same cloud shifted -> 0; row 1: skipped
    assert loss.item() < 1e-10
    loss.backward()
    assert torch.isfinite(est.grad).all()  # coincident points must not give NaN gradients
    est = torch.randn(2, 4, 3, requires_grad=True)
    assert_attached_zero(chamfer_loss(est, torch.randn(2, 5, 3), torch.zeros(2, 5, dtype=torch.bool)), est)


def test_uniformity_hand_cases():
    # Two antipodal unit vectors: |a - b|^2 = 4 -> log exp(-2 * 4) = -8; identical vectors -> log 1 = 0.
    z = torch.tensor([[3.0, 0.0], [-1.0, 0.0]], requires_grad=True)
    assert math.isclose(uniformity_loss(z).item(), -8.0, rel_tol=1e-6)
    collapsed = torch.ones(4, 3, requires_grad=True)
    loss = uniformity_loss(collapsed)
    assert abs(loss.item()) < 1e-6
    loss.backward()
    assert torch.isfinite(collapsed.grad).all()  # finite gradient exactly at collapse, where it matters
    one = torch.randn(1, 3, requires_grad=True)
    assert_attached_zero(uniformity_loss(one), one)
