"""Tests of the network modules and the TacFind aggregate, built from the real Hydra group YAML files."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from tacfind.networks.model import TacFind
from tacfind.networks.matcher import PointTokenizer
from tacfind.networks.point_transformer import VectorAttention, farthest_point_sample, gather, knn
from tacfind.spec import EnvSpec
from tacfind.train.losses import confidence_loss, contrastive_loss, uniformity_loss

torch.set_num_threads(4)
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
TOUCH_ENCODERS, MATCHERS = ["mlp", "cnn1d", "transformer"], ["points", "latent", "latent_head"]
SPEC = EnvSpec(obs_dim=24, priv_dim=5)
N_CLOUD = 192  # target cloud size in tests (>= the first transition-down stage of 128)


def make_cfg(touch_encoder="cnn1d", matcher="points", scale="small", history=10):
    """Root-like config from the real group files (the root config.yaml needs env/train groups not built yet)."""
    groups = dict(touch_encoder=touch_encoder, target_encoder="point_transformer", matcher=matcher, policy="mlp")
    return OmegaConf.merge(OmegaConf.load(CONFIGS / "scale" / f"{scale}.yaml"), {"history": history},
                           {g: OmegaConf.load(CONFIGS / g / f"{name}.yaml") for g, name in groups.items()})


def make_model(**kw) -> TacFind:
    torch.manual_seed(0)
    return TacFind(make_cfg(**kw), SPEC)


def make_inputs(B=4, H=10, valid_frac=0.1):
    L = H * SPEC.n_tips
    touch_pts = torch.cat([0.05 * torch.randn(B, L, 3), torch.nn.functional.normalize(torch.randn(B, L, 3), dim=-1),
                           5 * torch.rand(B, L, 1), torch.rand(B, L, 1)], -1)
    touch_mask = torch.rand(B, L) < valid_frac
    touch_mask[:, 0] = True
    cloud = torch.cat([0.05 * torch.randn(B, N_CLOUD, 3),
                       torch.nn.functional.normalize(torch.randn(B, N_CLOUD, 3), dim=-1)], -1)
    return dict(obs_hist=torch.randn(B, H, SPEC.obs_dim), hist_mask=torch.ones(B, H, dtype=torch.bool),
                touch_pts=touch_pts, touch_mask=touch_mask, cloud=cloud)


def grad_norm(module) -> float:
    return sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)


@pytest.mark.parametrize("history", [10, 30])
@pytest.mark.parametrize("matcher", MATCHERS)
@pytest.mark.parametrize("touch_encoder", TOUCH_ENCODERS)
def test_shapes(touch_encoder, matcher, history):
    model, B = make_model(touch_encoder=touch_encoder, matcher=matcher, history=history), 3
    x = make_inputs(B, history)
    dist, value, p = model(**x, priv=torch.randn(B, SPEC.priv_dim))
    dim = model.matcher.confidence.head.in_features
    assert dist.mean.shape == dist.stddev.shape == (B, SPEC.act_dim) and value.shape == (B,)
    assert p.z_touch.shape == p.z_diff.shape == p.e_touch.shape == (B, dim)
    assert p.conf_logit.shape == p.conf.shape == (B,) and ((p.conf >= 0) & (p.conf <= 1)).all()
    assert (p.est_points is not None) == (matcher == "latent_head")
    if p.est_points is not None:
        assert p.est_points.shape == (B, 8, 3)
    code = model.encode_target(x["cloud"])
    assert code.tokens.shape == (B, 64, dim) and code.pos.shape == (B, 64, 3) and code.z.shape == (B, dim)
    sub = code[torch.tensor([2, 0, 2, 1])]
    assert sub.tokens.shape == (4, 64, dim) and torch.equal(sub.z[0], code.z[2])


def test_scales_param_counts():
    counts = {s: make_model(scale=s).param_counts() for s in ["small", "base", "large"]}
    assert counts["small"]["total"] < counts["base"]["total"] < counts["large"]["total"]
    assert set(counts["base"]) >= {"touch_encoder", "target_encoder", "matcher", "policy", "matcher.embed",
                                   "matcher.confidence", "total"}
    for s in ["small", "large"]:  # the scale group also sizes the policy
        model, cfg = make_model(scale=s), make_cfg(scale=s)
        assert [m.out_features for m in model.policy.actor if isinstance(m, torch.nn.Linear)][:-1] == list(
            cfg.policy_hidden)


def test_grouped_attention_matches_naive_formula():
    """The memory-lean einsum path equals y_i = sum_j w_ij * (alpha(x_j) + theta_val(h_ij)) built per pair."""
    torch.manual_seed(0)
    attn, x, pos = VectorAttention(16, groups=4, rel_dim=8), torch.randn(2, 7, 16), torch.randn(2, 7, 3)
    mask = torch.rand(2, 7) > 0.3
    for idx in [knn(pos, pos, 4, mask), None]:
        nb = (lambda t: t[:, None]) if idx is None else (lambda t: gather(t, idx))
        h = attn.theta(pos[:, :, None] - nb(pos))
        logits = attn.gamma(attn.phi(x)[:, :, None] - nb(attn.psi(x)) + attn.theta_rel(h))
        valid = nb(mask)[..., None]
        w = logits.masked_fill(~valid, -1e30).softmax(2) * valid  # (B, Q, K, G)
        naive = (w.repeat_interleave(4, -1) * (nb(attn.alpha(x)) + attn.theta_val(h))).sum(2)
        assert torch.allclose(attn(x, x, pos, pos, idx, mask), naive, atol=1e-5)
    y = attn(x, x, kv_mask=mask)  # no positions: no positional term at all
    w = attn.gamma(attn.phi(x)[:, :, None] - attn.psi(x)[:, None]).masked_fill(~mask[:, None, :, None], -1e30)
    assert torch.allclose(y, (w.softmax(2).repeat_interleave(4, -1) * attn.alpha(x)[:, None]).sum(2), atol=1e-5)


def test_masked_fps_and_token_cap():
    pos, mask = torch.randn(2, 10, 3), torch.zeros(2, 10, dtype=torch.bool)
    mask[0, [1, 4, 7]] = True
    mask[1, :6] = True
    idx = farthest_point_sample(pos, 8, mask, start=torch.tensor([4, 2]))
    assert idx[0, 0] == 4 and idx[1, 0] == 2 and all(len(set(row.tolist())) == 8 for row in idx)
    assert set(idx[0, :3].tolist()) == {1, 4, 7} and mask[1, idx[1, :6]].all()  # valid points come first
    tok = PointTokenizer(16, 1, max_tokens=4)
    pts = torch.rand(2, 10, 8)
    tokens, tmask, _ = tok(None, pts, mask)
    assert tokens.shape == (2, 4, 16) and tmask[0].tolist() == [True] * 3 + [False] and tmask[1].all()
    tokens, tmask, _ = tok(None, pts, mask & (torch.arange(10) < 1))  # row 0: no contact, row 1: one
    assert tokens.shape == (2, 1, 16) and tmask.tolist() == [[False], [True]]


def test_vector_attention_all_masked_gives_zeros():
    attn, x = VectorAttention(16), torch.randn(2, 5, 16, requires_grad=True)
    mask = torch.tensor([[False] * 5, [True, False, True, False, False]])
    y = attn(x, x, kv_mask=mask)
    assert torch.equal(y[0], torch.zeros(5, 16)) and torch.isfinite(y).all()
    y.sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("matcher", MATCHERS)
@pytest.mark.parametrize("touch_encoder", TOUCH_ENCODERS)
def test_masking_finite_and_padding_ignored(touch_encoder, matcher):
    model, H = make_model(touch_encoder=touch_encoder, matcher=matcher), 10
    x = make_inputs(4, H)
    x["touch_mask"][:] = False  # no contact at all
    x["hist_mask"][:, : H // 2] = False  # episode started half a window ago
    x["obs_hist"][:, : H // 2] = 0.0
    dist, value, p = model(**x, priv=torch.zeros(4, SPEC.priv_dim), detach_matcher=False)
    outs = [dist.mean, value, p.z_touch, p.z_diff, p.conf_logit, p.e_touch]
    assert all(torch.isfinite(t).all() for t in outs)
    sum(t.sum() for t in outs).backward()
    assert all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None)
    # Garbage in padded history steps and invalid touch points must not change anything.
    y = dict(x, obs_hist=x["obs_hist"].clone(), touch_pts=x["touch_pts"].clone())
    y["obs_hist"][:, : H // 2] = 7.0
    y["touch_pts"] += 1.0
    priv = torch.zeros(4, SPEC.priv_dim)
    with torch.no_grad():
        assert torch.allclose(model(**x, priv=priv)[2].z_diff, model(**y, priv=priv)[2].z_diff, atol=1e-5)


def test_target_encoder_translation_invariant():
    model, cloud = make_model(), make_inputs(2)["cloud"]
    moved = cloud.clone()
    moved[..., :3] += torch.tensor([0.3, -1.2, 0.7])
    with torch.no_grad():
        a, b = model.encode_target(cloud), model.encode_target(moved)
    assert torch.allclose(a.z, b.z, atol=1e-4) and torch.allclose(a.tokens, b.tokens, atol=1e-4)


@pytest.mark.parametrize("matcher", MATCHERS)
def test_gradient_routing(matcher):
    model, x = make_model(matcher=matcher), make_inputs(4)
    lib = model.encode_target(x["cloud"])
    p = model.perceive(x["obs_hist"], x["hist_mask"], x["touch_pts"], x["touch_mask"], lib)
    loss, _ = contrastive_loss(p.e_touch, lib.z, torch.tensor([0, 1, -1, 3]), 0.1)
    loss.backward(retain_graph=True)
    assert grad_norm(model.target_encoder) > 0 and grad_norm(model.matcher.embed) > 0
    assert grad_norm(model.policy) == 0 and grad_norm(model.matcher.confidence) == 0
    model.zero_grad()
    confidence_loss(p.conf_logit, torch.tensor([1.0, 0.0, 0.5, 0.2])).backward()
    assert grad_norm(model.matcher.confidence) > 0 and grad_norm(model.policy) == 0


def test_detach_matcher():
    model, x = make_model(), make_inputs(4)
    for detach in [True, False]:
        model.zero_grad()
        dist, value, _ = model(**x, priv=torch.randn(4, SPEC.priv_dim), detach_matcher=detach)
        (dist.mean.sum() + value.sum()).backward()
        assert grad_norm(model.touch_encoder) > 0 and grad_norm(model.policy) > 0  # z_touch keeps its path
        assert (grad_norm(model.matcher) == 0) == detach and (grad_norm(model.target_encoder) == 0) == detach


def test_set_trainable():
    model = make_model()
    params = model.set_trainable(["target_encoder", "matcher.embed"])
    trainable = {id(q) for m in [model.target_encoder, model.matcher.embed] for q in m.parameters()}
    assert {id(q) for q in params} == trainable
    assert all(q.requires_grad == (id(q) in trainable) for q in model.parameters())
    assert model.matcher.embed.training and model.target_encoder.training
    assert not model.policy.training and not model.matcher.confidence.training and not model.touch_encoder.training
    for mode in [False, True]:  # robust to call order: train() keeps the frozen parts in eval mode
        model.train(mode)
        assert model.matcher.embed.training == mode and model.target_encoder.training == mode
        assert not model.policy.training and not model.matcher.confidence.training
        assert not model.touch_encoder.training and not model.policy.actor[0].training


# ---------------------------------------------------------------- learnability on synthetic primitives


def sample_primitive(kind: str, n: int) -> torch.Tensor:
    """(n, 6) [xyz (m), unit outward normal] on a primitive surface, area-uniform."""
    if kind == "sphere":  # radius 1.5 cm
        nrm = torch.nn.functional.normalize(torch.randn(n, 3), dim=-1)
        return torch.cat([0.015 * nrm, nrm], -1)
    if kind == "cylinder":  # radius 1 cm, height 10 cm, side only (caps are small)
        phi, z = 2 * torch.pi * torch.rand(n), 0.1 * torch.rand(n) - 0.05
        nrm = torch.stack([phi.cos(), phi.sin(), torch.zeros(n)], -1)
        return torch.cat([0.01 * nrm + torch.stack([0 * z, 0 * z, z], -1), nrm], -1)
    half = torch.tensor({"cube": [0.015, 0.015, 0.015], "plate": [0.06, 0.04, 0.005]}[kind])
    area = torch.stack([half[1] * half[2], half[0] * half[2], half[0] * half[1]]).repeat(2)
    face = torch.multinomial(area, n, replacement=True)  # 0..5: axis = face % 3, sign = +/- by face // 3
    axis, sign = face % 3, 1.0 - 2.0 * (face // 3).float()
    xyz = (2 * torch.rand(n, 3) - 1) * half
    xyz[torch.arange(n), axis] = sign * half[axis]
    return torch.cat([xyz, torch.nn.functional.one_hot(axis, 3) * sign[:, None]], -1)


def random_rotation(B: int) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(B, 3, 3))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))[:, None]
    return q * torch.linalg.det(q)[:, None, None]  # proper rotation (det = +1)


def touch_batch(lib: torch.Tensor, B: int, patch=16, L=24):
    """Random surface patches of random library objects under a random rotation + translation, padded to L."""
    obj = torch.randint(len(lib), (B,))
    pts = lib[obj]  # (B, N, 6)
    centre = pts[torch.arange(B), torch.randint(pts.shape[1], (B,)), :3]
    idx = (pts[..., :3] - centre[:, None]).norm(dim=-1).topk(patch, largest=False).indices
    patch_pts = torch.gather(pts, 1, idx[..., None].expand(-1, -1, 6))
    rot, shift = random_rotation(B), 0.3 * torch.randn(B, 1, 3)
    xyz, nrm = patch_pts[..., :3] @ rot.transpose(1, 2) + shift, patch_pts[..., 3:] @ rot.transpose(1, 2)
    touch_pts = torch.zeros(B, L, 8)
    touch_pts[:, :patch] = torch.cat([xyz, nrm, 3 * torch.rand(B, patch, 1), torch.rand(B, patch, 1)], -1)
    touch_mask = torch.zeros(B, L, dtype=torch.bool)
    touch_mask[:, :patch] = True
    return touch_pts, touch_mask, obj


def test_contrastive_learnability():
    torch.manual_seed(0)
    model = TacFind(make_cfg(matcher="points"), SPEC)
    lib = torch.stack([sample_primitive(k, 160) for k in ["sphere", "cylinder", "cube", "plate"]])
    opt = torch.optim.Adam(model.set_trainable(["target_encoder", "matcher.embed"]), lr=2e-3)
    eval_pts, eval_mask, eval_obj = touch_batch(lib, 128)
    obs_hist, hist_mask = torch.zeros(1, 10, SPEC.obs_dim), torch.ones(1, 10, dtype=torch.bool)

    def step(pts, mask, obj):
        code = model.encode_target(lib)
        z_touch = model.touch_encoder(obs_hist, hist_mask).expand(len(obj), -1)  # unused by the `points` tokenizer
        e_touch = model.matcher.embed(z_touch, pts, mask)[2]
        loss, acc = contrastive_loss(e_touch, code.z, obj, temperature=0.1)
        # Without the uniformity term the library embeddings collapse onto one vector (the optimum while e_touch is
        # still uninformative) and training stalls at chance.
        return loss + uniformity_loss(code.z), acc

    with torch.no_grad():
        acc_before = step(eval_pts, eval_mask, eval_obj)[1].item()
    for _ in range(120):
        loss, _ = step(*touch_batch(lib, 64))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc_after = step(eval_pts, eval_mask, eval_obj)[1].item()
    print(f"learnability: top-1 accuracy {acc_before:.3f} -> {acc_after:.3f} (chance 0.25)")
    assert acc_after > 0.5, (acc_before, acc_after)
