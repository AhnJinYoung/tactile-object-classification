"""BinEnv on the CPU reference backend (fast), plus MuJoCo Warp checks on its CPU device (marked slow)."""

import pytest
import torch

from tacfind.envs.bin_env import GROUP_SIZES, BinEnv
from tacfind.envs.explorer import Explorer
from tacfind.envs.sim import CpuSim

N = 4


@pytest.fixture
def env(cfg):
    cfg.obs_groups = list(GROUP_SIZES)  # every group, to check all sizes
    e = BinEnv(cfg, N, history=30, device="cpu", seed=0)
    yield e
    e.sim.pool.shutdown()


def random_actions(n: int, steps: int, seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(n, 28, generator=g) * 2 - 1 for _ in range(steps)]


def test_shapes_sizes_and_ranges(env):
    obs = env.reset()
    assert env.spec.obs_dim == sum(GROUP_SIZES.values()) == obs["o"].shape[1]
    assert env.spec.priv_dim == 10 * env.cfg.n_slots and env.clouds.shape[1:] == (env.cfg.objects.n_points, 6)
    for a in random_actions(N, 30):
        obs, reward, done, info = env.step(a)
        assert obs["o"].shape == (N, env.spec.obs_dim) and obs["o"].dtype == torch.float32
        assert obs["pts"].shape == (N, 5, 7) and obs["pts_valid"].shape == (N, 5)
        assert obs["pts_valid"].dtype == torch.bool
        assert obs["first"].shape == (N,) and obs["target"].dtype == torch.long
        assert torch.isfinite(obs["o"]).all() and torch.isfinite(reward).all() and reward.shape == (N,)
        assert info["tip_obj"].shape == (N, 5) and info["priv"].shape == (N, env.spec.priv_dim)
        lo, hi = env.cfg.n_active
        assert ((info["n_objects"] >= lo) & (info["n_objects"] <= hi)).all()
        empty = info["priv"].view(N, env.cfg.n_slots, -1)[torch.arange(env.cfg.n_slots) >= info["n_objects"][:, None]]
        assert (empty == 0).all()  # padded slots are zeros
        assert ((info["conf_gt"] >= 0) & (info["conf_gt"] <= 1)).all()
        lib = set(env.lib_ids.tolist()) | {-1}
        assert set(info["touched"].tolist()) <= lib and set(info["tip_obj"].flatten().tolist()) <= lib
        thr = env.randomizer.threshold[:, None].expand(N, 5)
        assert (obs["pts"][..., 6][obs["pts_valid"]] >= thr[obs["pts_valid"]] - 1e-6).all()
        assert torch.allclose(obs["pts"][..., 3:6][obs["pts_valid"]].norm(dim=-1), torch.ones(1), atol=1e-4)
        assert (obs["pts"][~obs["pts_valid"]] == 0).all()
    assert env.sim.health()["overflow"] == 0


def test_same_seed_same_resets_and_trajectory(cfg):
    runs = []
    for _ in range(2):
        e = BinEnv(cfg, 2, history=30, device="cpu", seed=3)
        obs = [e.reset()["o"]] + [e.step(a)[0]["o"] for a in random_actions(2, 5)]
        runs.append((e.bank_pose, e.active.clone(), e.target.clone(), torch.stack(obs)))
        e.sim.pool.shutdown()
    for a, b in zip(*runs):
        assert torch.equal(a, b)  # the CPU backend is deterministic; Warp only promises equal reset states


def test_auto_reset(env):
    env.max_steps = 3
    env.reset()
    for k, a in enumerate(random_actions(N, 4)):
        obs, _, done, info = env.step(a)
        assert done.all() == (k == 2) and info["timeout"].all() == (k == 2)
        assert torch.equal(obs["first"], done)
    assert (env.t == 1).all()


def test_override_and_world_state(env):
    obs = env.reset()
    env.override(1, torch.full((28,), 0.5))
    env.step(torch.zeros(N, 28))
    assert torch.allclose(env.prev_action[1], torch.full((28,), 0.5)) and (env.prev_action[0] == 0).all()
    env.override(1, None)
    st = env.world_state(1)
    assert st["qpos"].shape == (env.mjm.nq,) and st["pts"].shape == (5, 3) and st["active_lib"].shape == (4,)
    assert int(st["target_lib"]) in env.lib_ids.tolist()
    assert st["q_target"].shape == (28,) and st["qvel"].shape == (env.mjm.nv,)


def test_reset_worlds_and_perturb(env):
    env.reset()
    for _ in range(3):
        env.step(torch.zeros(N, 28))
    mask = torch.tensor([True] + [False] * (N - 1))
    env.reset_worlds(mask)
    assert env.t[0] == 0 and (env.t[1:] == 3).all()
    body = int(env.obj_body[env.active[2, 0]])
    z0 = float(env.sim.xpos[2, body, 2])
    wrench = torch.zeros(env.mjm.nbody, 6)
    wrench[body, 2] = 2 * float(env.obj_weight[env.active[2, 0]])  # twice the weight: rises at about 1 g
    env.perturb(2, wrench)
    for _ in range(2):  # 0.1 s: ~5 cm, still inside the bin (leaving it would end the episode)
        env.step(torch.zeros(N, 28))
    assert float(env.sim.xpos[2, body, 2]) > z0 + 0.01
    env.perturb(2, None)
    assert (env.xfrc_extra == 0).all()


def test_scripted_grasp_lifts_a_ball(cfg):
    import numpy as np

    from tacfind.envs.bin_env import plain
    from tacfind.envs.grasp_check import LiftTrial
    from tacfind.envs.objects import build_library

    c = plain(cfg)
    ball = next(o for o in build_library(c.objects) if o.family == "golf_ball")
    ex, close = c.explorer, np.asarray(c.explorer.close)
    out = LiftTrial(c, ball).run(np.random.default_rng(0), ex.grasp_pre * close, close, ex.grasp_tilt, ex.grasp_shift)
    assert out["success"], out


def test_domain_randomisation_per_world(env):
    env.reset()
    body = env.obj_body
    masses = torch.stack([torch.as_tensor(m.body_mass[body.numpy()]) for m in env.sim.models])  # (N, n_obj)
    nominal = torch.as_tensor(env.mjm.body_mass[body.numpy()])
    lo, hi = env.cfg.randomize.mass_scale
    assert ((masses / nominal >= lo - 1e-6) & (masses / nominal <= hi + 1e-6)).all()
    assert not torch.allclose(masses[0], masses[1])  # each world draws its own
    t_lo, t_hi = env.cfg.randomize.force_threshold
    assert ((env.randomizer.threshold >= t_lo) & (env.randomizer.threshold <= t_hi)).all()
