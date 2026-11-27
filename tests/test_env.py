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
        assert ((info["conf_gt"] >= 0) & (info["conf_gt"] <= 1)).all()
        lib = set(env.lib_ids.tolist()) | {-1}
        assert set(info["touched"].tolist()) <= lib and set(info["tip_obj"].flatten().tolist()) <= lib
        assert (obs["pts"][..., 6][obs["pts_valid"]] >= env.cfg.force_threshold).all()
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


def test_explorer_touches_objects(cfg):
    e = BinEnv(cfg, N, history=30, device="cpu", seed=1)
    ex = Explorer(e, seed=0)
    assert ex.ik_error < 1e-3
    obs, contact, conf = e.reset(), 0, 0.0
    for _ in range(150):
        obs, reward, _, info = e.step(ex.act(obs))
        assert torch.isfinite(obs["o"]).all() and torch.isfinite(reward).all()
        contact += int(obs["pts_valid"].any(1).sum())
        conf = max(conf, float(info["conf_gt"].max()))
    assert contact > 0 and conf > 0 and e.sim.health()["overflow"] == 0
    e.sim.pool.shutdown()


@pytest.mark.slow
def test_warp_matches_cpu_backend(cfg):
    """Same model, state, controls and applied forces, 40 substeps (hand in the air, objects resting): close."""
    from tacfind.envs.sim import WarpSim

    e = BinEnv(cfg, 1, history=30, device="cpu", seed=0)
    e.reset()
    qpos, qvel = e.sim.qpos.clone(), torch.zeros(1, e.mjm.nv)
    sims = [CpuSim(e.mjm, 1), WarpSim(e.mjm, 1, "cpu", nconmax=64, njmax=400)]
    for s in sims:
        s.set_state(torch.ones(1, dtype=torch.bool), qpos, qvel)
        s.set_ctrl(e.q_target + torch.linspace(-0.05, 0.05, 28))  # a small move of every joint
        s.set_xfrc(e.sim.xfrc_applied)
        s.step(40)
    cpu, warp = (s.qpos.cpu() for s in sims)
    assert torch.allclose(cpu[:, e.qadr], warp[:, e.qadr], atol=2e-3)  # robot joints, rad
    obj = (e.obj_qadr[e.active[0]][:, None] + torch.arange(3)).flatten()
    assert torch.allclose(cpu[:, obj], warp[:, obj], atol=2e-3)  # active object positions, m
    e.sim.pool.shutdown()


@pytest.mark.slow
def test_warp_env_end_to_end(cfg):
    cfg.sim, cfg.bank_size, cfg.settle_time = "warp", 2, 0.5
    e = BinEnv(cfg, 2, history=30, device="cpu", seed=0)
    ex = Explorer(e, seed=0)
    obs = e.reset()
    for _ in range(10):
        obs, reward, done, info = e.step(ex.act(obs))
        assert torch.isfinite(obs["o"]).all() and torch.isfinite(reward).all()
        assert obs["o"].shape == (2, e.spec.obs_dim) and info["conf_gt"].shape == (2,)
    assert e.sim.health()["overflow"] == 0
