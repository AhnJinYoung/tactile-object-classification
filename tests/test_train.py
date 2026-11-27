"""Trainer tests: history windows vs a brute-force reconstruction, GAE by hand, and every stage of every scheme for one
iteration on the CPU reference backend with 2 worlds (losses finite, the right modules learn, run-directory files)."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

from tacfind.train.logger import RunLogger  # noqa: E402
from tacfind.train.rollout import Rollout, gae  # noqa: E402
from tacfind.train.stages import parse_stages  # noqa: E402
from tacfind.train.trainer import Trainer  # noqa: E402

torch.set_num_threads(4)
REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------------------------------------- windows --
def test_windows_match_brute_force():
    """Two iterations with episode boundaries inside and across them: indexed windows == per-world reconstruction."""
    N, T, H, n_lib, D = 3, 6, 4, 5, 2
    g = torch.Generator().manual_seed(0)
    buf, timeline = Rollout(N, T, H, n_lib, torch.device("cpu")), [[] for _ in range(N)]
    firsts = {0: [0, 1, 2], 3: [1], 5: [0], 7: [2], 8: [1], 10: [0, 2]}  # global step -> worlds starting an episode

    def add(k, step):
        first = torch.zeros(N, dtype=torch.bool)
        first[firsts.get(step, [])] = True
        f = dict(o=torch.randn(N, D, generator=g), pts=torch.randn(N, 5, 7, generator=g),
                 valid=torch.rand(N, 5, generator=g) < 0.1, first=first, target=torch.randint(n_lib, (N,), generator=g),
                 obj=torch.randint(-1, n_lib, (N, 5), generator=g), conf_gt=torch.rand(N, generator=g),
                 priv=torch.randn(N, 3, generator=g))
        buf.add_obs(k, **f)
        for n in range(N):
            timeline[n].append({key: v[n] for key, v in f.items()})

    def brute(n, step):
        start = max(s for s in range(step + 1) if timeline[n][s]["first"])
        rows = []
        for j in range(H):
            s = step - H + 1 + j
            ok = s >= start
            e = timeline[n][s] if ok else None
            age = (H - 1 - j) / (H - 1)
            valid = e["valid"] & ok if ok else torch.zeros(5, dtype=torch.bool)
            pts = torch.cat([e["pts"], torch.full((5, 1), age)], -1) * valid[:, None] if ok else torch.zeros(5, 8)
            rows.append((e["o"] if ok else torch.zeros(D), ok, pts, valid, e["obj"] if ok else None))
        objs = Counter(int(o) for _, ok, _, v, obj in rows if ok for o, vv in zip(obj, v) if vv and o >= 0)
        top = max(objs.values()) if objs else 0
        touched = min(o for o, c in objs.items() if c == top) if objs else -1
        return (torch.stack([r[0] for r in rows]), torch.tensor([r[1] for r in rows]),
                torch.cat([r[2] for r in rows]), torch.cat([r[3] for r in rows]), touched, timeline[n][step])

    add(T + H - 1, 0)  # the "next observation" slot; advance() moves it to step 0
    step = 0
    for _ in range(2):
        buf.advance()
        for t in range(T):
            step += 1
            add(t + H, step)
        t_idx, n_idx = torch.arange(T + 1).repeat_interleave(N), torch.arange(N).repeat(T + 1)
        w = buf.window(t_idx, n_idx)
        for b, (t, n) in enumerate(zip(t_idx.tolist(), n_idx.tolist())):
            o, hm, pts, tm, touched, last = brute(n, step - T + t)
            assert torch.equal(w["obs_hist"][b], o) and torch.equal(w["hist_mask"][b], hm)
            assert torch.allclose(w["touch_pts"][b], pts) and torch.equal(w["touch_mask"][b], tm)
            assert int(w["touched"][b]) == touched and w["conf_gt"][b] == last["conf_gt"]
            assert w["target"][b] == last["target"] and torch.equal(w["priv"][b], last["priv"])
        has = buf.has_contact()
        assert torch.equal(has, w["touch_mask"].any(1).view(T + 1, N)[:T]) and has.any() and not has.all()
    assert not w["hist_mask"].all() and w["hist_mask"].any()  # the case covers masked and unmasked steps


def test_gae_by_hand():
    reward, value = torch.tensor([[1.0], [2.0], [3.0]]), torch.full((3, 1), 0.5)
    done = torch.tensor([[False], [True], [False]])
    adv, ret = gae(reward, value, done, torch.tensor([1.0]), gamma=0.9, lam=0.8)
    # t=2: 3 + 0.9 * 1 - 0.5 = 3.4; t=1 (terminal): 2 - 0.5 = 1.5; t=0: 1 + 0.45 - 0.5 + 0.72 * 1.5 = 2.03
    assert torch.allclose(adv[:, 0], torch.tensor([2.03, 1.5, 3.4])) and torch.allclose(ret, adv + 0.5)


# ------------------------------------------------------------------------------------------------- stages --
SMALL = ["env.sim=cpu", "env.bank_size=4", "n_envs=2", "rollout_steps=8", "history=10", "scale=small",
         "optim.batch_size.perception=8", "optim.batch_size.ppo=8", "optim.epochs=1", "max_iters=1", "viz_every=1", "viz_steps=3", "perceive_chunk=1",
         "device=cpu", "env.objects.n_points=192"]


@pytest.fixture(autouse=True)
def assets():
    if not (REPO / "third_party" / "sharpa").is_dir():
        pytest.skip("robot assets missing: run scripts/fetch_assets.py")


def make_cfg(tmp_path, *overrides):
    with initialize_config_dir(str(REPO / "configs"), version_base="1.3"):
        return compose("config", [*SMALL, f"runs_dir={tmp_path}", *overrides])


def snapshot(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def changed(before, after, prefix):
    return any(not torch.equal(before[k], after[k]) for k in before if k.startswith(f"{prefix}."))


def run_checked(cfg) -> tuple[Trainer, RunLogger]:
    """Run every stage; after each, the trained modules changed and all others did not."""
    logger = RunLogger(cfg)
    tr = Trainer(cfg, logger)
    warm = parse_stages(cfg.train, tr.model, cfg.optim)[0]
    for _ in range(25):  # explorer steps until some windows hold contacts (the hand needs a few s to touch)
        tr.collect(warm, 0)
        if tr.buf.pos["valid"][-1].any():
            break
    assert tr.buf.pos["valid"].any()
    sampled, losses = [], tr.losses

    def recording_losses(stage, t, n, adv, ret):  # which windows each stage trains on
        sampled.append((len(t), bool(tr.buf.has_contact()[t, n].all())))
        return losses(stage, t, n, adv, ret)

    tr.losses = recording_losses
    for stage in parse_stages(cfg.train, tr.model, cfg.optim, max_iters=cfg.max_iters):
        before = snapshot(tr.model)
        sampled.clear()
        tr.run_stage(stage)
        after = snapshot(tr.model)
        n_contact = int(tr.buf.has_contact().sum())
        if stage.contact_only:  # contrastive (+ uniform, chamfer): only windows that hold a touch
            assert all(ok for _, ok in sampled) and sum(k for k, _ in sampled) == n_contact > 0, stage.name
        else:  # confidence / PPO: every window, in minibatches of at most batch_size
            assert sum(k for k, _ in sampled) == cfg.rollout_steps * cfg.n_envs * stage.epochs, stage.name
        assert max(k for k, _ in sampled) <= stage.batch_size
        for name in ["touch_encoder", "target_encoder", "matcher.embed", "matcher.confidence", "policy"]:
            trained = any(name == s or name.startswith(f"{s}.") for s in stage.train)
            assert changed(before, after, name) == trained, (stage.name, name)
    logger.close("done")
    tr.env.sim.pool.shutdown()
    return tr, logger


def check_run_dir(run: Path, stages: list[str]):
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert {r["stage"] for r in rows} == set(stages)
    for r in rows:
        assert {"step", "iter", "stage", "time", "rollout/contact_frac", "rollout/contact_window_frac",
                "perf/env_steps_per_s"} <= set(r)
        assert all(v is None or abs(v) < 1e9 for k, v in r.items() if "/" in k)  # finite (NaN -> null)
    summary = json.loads((run / "summary.json").read_text())
    assert summary["status"] == "done" and summary["params"]["total"] > 0 and summary["env_steps"] > 0
    assert set(summary["stages"]) == set(stages)
    for s in stages:
        assert (run / "ckpt" / f"{s}.pt").is_file() and list((run / "videos").glob(f"{s}_*.mp4"))
    assert (run / "ckpt" / "last.pt").is_file() and (run / "figures" / "curves.png").is_file()
    assert (run / "config.yaml").is_file() and (run / "git.txt").is_file() and list((run / "tb").iterdir())
    return rows


@pytest.mark.parametrize("matcher", ["points", "latent", "latent_head"])
def test_staged(tmp_path, matcher):
    cfg = make_cfg(tmp_path, f"matcher={matcher}", f"name=staged_{matcher}")
    tr, logger = run_checked(cfg)
    rows = check_run_dir(logger.dir, ["contrastive", "confidence", "encoder_ppo", "policy_ppo"])
    by_stage = {r["stage"]: r for r in rows}
    assert by_stage["contrastive"]["loss/uniform"] is not None and by_stage["confidence"]["conf/mae"] is not None
    assert by_stage["encoder_ppo"]["loss/policy"] is not None and by_stage["policy_ppo"]["ppo/entropy"] is not None
    for key in ["loss/contrastive", "loss/chamfer", "loss/confidence", "loss/policy", "loss/value"]:
        assert all(r[key] is not None for r in rows if key in r)
    if matcher != "points":  # the alias touch_side includes the touch encoder for these matchers
        assert "touch_encoder" in parse_stages(cfg.train, tr.model, cfg.optim)[0].train
    if matcher == "points":  # a second run starts from the first one's weights and runs only the PPO stages
        ckpt = torch.load(logger.dir / "ckpt" / "last.pt")
        cfg2 = make_cfg(tmp_path, f"init_from={logger.dir}", "run_stages=[policy_ppo]", "name=second")
        logger2 = RunLogger(cfg2)
        tr2 = Trainer(cfg2, logger2)
        assert all(torch.equal(v, ckpt["model"][k]) for k, v in tr2.model.state_dict().items())
        tr2.run()
        logger2.close("done")
        tr2.env.sim.pool.shutdown()
        check_run_dir(logger2.dir, ["policy_ppo"])
        evaluate_script(tmp_path, logger.dir)


def evaluate_script(tmp_path, run: Path):
    """scripts/evaluate.py on the held-out objects: one eval row with finished episodes, a video, status done."""
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "evaluate.py"), "--run", str(run), "--episodes", "2",
                          "--split", "test", "env.timeout=0.4", "viz_steps=3", "perceive_chunk=4"],
                         cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    (ev,) = tmp_path.glob("*_eval_*_test_policy")
    (row,) = [json.loads(line) for line in (ev / "metrics.jsonl").read_text().splitlines()]
    assert row["stage"] == "eval" and row["rollout/episodes"] >= 2 and row["rollout/success_rate"] is not None
    assert row["conf/mae"] is not None and (ev / "videos" / "eval_policy.mp4").is_file()
    assert json.loads((ev / "summary.json").read_text())["status"] == "done"


def test_rollout_logp_equals_update_logp(tmp_path):
    """The window used when acting is the one rebuilt in the update: same weights -> same log-probabilities."""
    cfg = make_cfg(tmp_path, "train=joint")
    tr = Trainer(cfg, RunLogger(cfg))
    stage = parse_stages(cfg.train, tr.model, cfg.optim)[0]
    tr.model.set_trainable(stage.train)
    tr.collect(stage, 0)
    tr.collect(stage, 1)  # the second iteration's windows reach back into the first one
    t, n = torch.arange(cfg.rollout_steps).repeat_interleave(2), torch.arange(2).repeat(cfg.rollout_steps)
    with torch.no_grad():
        w = tr.buf.window(t, n)
        p = tr.model.perceive(w["obs_hist"], w["hist_mask"], w["touch_pts"], w["touch_mask"],
                              tr.model.encode_target(tr.clouds)[tr.lib_pos[w["target"]]])
        dist, value = tr.model.act(w["obs_hist"], p, w["priv"])
    st = tr.buf.step
    assert torch.allclose(dist.log_prob(st["action"][t, n]).sum(-1), st["logp"][t, n], atol=1e-3)
    assert torch.allclose(value, st["value"][t, n], atol=1e-4)
    tr.env.sim.pool.shutdown()


def test_joint_via_hydra_cli(tmp_path):
    """scripts/train.py end to end: Hydra keeps the working directory, the run lands in runs_dir."""
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "train.py"), *SMALL, f"runs_dir={tmp_path}",
                          "train=joint", "name=joint"], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    (run,) = tmp_path.glob("*_joint")
    rows = check_run_dir(run, ["joint"])
    assert all(rows[-1][k] is not None for k in ["loss/policy", "loss/contrastive", "loss/confidence", "ppo/kl"])
    summary = json.loads((run / "summary.json").read_text())
    assert summary["choices"]["train"] == "joint" and summary["choices"]["experiment"] is None
    assert "train=joint" in summary["overrides"]
    assert sorted(p.name for p in tmp_path.iterdir()) == [run.name]  # Hydra wrote nothing else


def test_experiment_presets():
    """Every preset composes, names itself after its file, and has a one-line description."""
    for path in sorted((REPO / "configs" / "experiment").glob("*.yaml")):
        with initialize_config_dir(str(REPO / "configs"), version_base="1.3"):
            cfg = compose("config", [f"+experiment={path.stem}"])
        assert cfg.name == path.stem and cfg.description and "\n" not in cfg.description, path.stem
    with initialize_config_dir(str(REPO / "configs"), version_base="1.3"):
        cfg = compose("config", ["+experiment=E004_matcher_latent_head"])
    assert cfg.matcher.tokenizer._target_.endswith("LatentHeadTokenizer")


def test_live_view_spectates_only(tmp_path, monkeypatch):
    """live_port set: one viewer sync per env step; a browser push, driver switch, reset or pause changes nothing."""
    from types import SimpleNamespace

    from tacfind.viz import live

    syncs = []

    def fake_start(self):  # stands in for the web server
        self.handle = SimpleNamespace(sync=lambda m, d: syncs.append(1), send_to_viewer=lambda msg: None,
                                      is_running=lambda: True, close=lambda: None)
        self.port = 0
        return self

    monkeypatch.setattr(live.LiveViewer, "start", fake_start)
    cfg = make_cfg(tmp_path, "live_port=0", "viz_world=1")
    tr = Trainer(cfg, RunLogger(cfg))
    pushes = []
    tr.env.perturb = lambda w, f: pushes.append(w)
    bridge = tr.live.bridge
    bridge.xfrc[:] = 1.0  # the browser user drags a body ...
    bridge.cmd.driver, bridge.cmd.paused, bridge.cmd.resets = "manual", True, [0]  # ... and asks for control
    t_before = tr.env.t.clone()
    tr.collect(parse_stages(cfg.train, tr.model, cfg.optim)[0], 0)
    assert len(syncs) == cfg.rollout_steps and tr.live.world == 1
    assert not pushes and not tr.env.override_mask.any() and (tr.env.t - t_before == cfg.rollout_steps).all()
    tr.close()
