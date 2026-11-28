"""Report page (tacfind/report, docs/report/src): module specs against the real tensors, the export round trip
through Hydra, and the page build from run directories (fake runs in a temporary directory)."""

import json
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from tacfind.networks.model import TacFind
from tacfind.report.configs import CONFIGS, compose_preset, config_sources
from tacfind.report.page import build
from tacfind.report.runs import downsample, parse_overrides
from tacfind.report.specs import active_modules, expand_steps, load_specs, shape_numbers, symbol_values
from tacfind.spec import EnvSpec

REPO = Path(__file__).resolve().parents[1]
PRESETS = sorted(p.stem for p in (CONFIGS / "experiment").glob("*.yaml"))
NODE = shutil.which("node")
B = 2


def dummy_batch(cfg: dict, sym: dict) -> tuple:
    """Inputs of TacFind.forward with every step and contact real (so T = min(L, max_tokens))."""
    H, L = sym["H"], sym["L"]
    normal = torch.nn.functional.normalize(torch.randn(B, L, 3), dim=-1)
    touch_pts = torch.cat([0.02 * torch.randn(B, L, 3), normal, torch.rand(B, L, 1) + 0.5, torch.rand(B, L, 1)], -1)
    cloud_xyz = 0.03 * torch.randn(B, sym["N"], 3)
    cloud = torch.cat([cloud_xyz, torch.nn.functional.normalize(cloud_xyz, dim=-1)], -1)
    return (torch.randn(B, H, sym["D_o"]), torch.ones(B, H, dtype=torch.bool), touch_pts,
            torch.ones(B, L, dtype=torch.bool), cloud, torch.randn(B, sym["P"]))


@pytest.mark.parametrize("preset", PRESETS)
def test_spec_shapes_match_the_networks(preset):
    """Every step shape of every active module spec equals the traced tensor shape of the real model."""
    torch.manual_seed(0)
    cfg, choices = compose_preset(preset)
    specs = load_specs()
    sym = symbol_values(cfg, choices, specs)
    model = TacFind(OmegaConf.create(cfg), EnvSpec(obs_dim=sym["D_o"], act_dim=sym["A"], priv_dim=sym["P"]))
    model.train()  # no fused inference path: every submodule is called, so every hook fires
    seen, steps = {}, []
    for module in active_modules(choices, specs).values():
        steps += expand_steps(module, cfg, sym)
    for step in steps:
        t = step["trace"]
        if "module" not in t:
            continue
        index = t.get("index", 0)
        key, mod = (t["module"], t["io"], index), model.get_submodule(t["module"])
        def record(key, x, i):  # returns None: a hook's return value would replace the module's input / output
            seen.setdefault(key, tuple((x[i] if isinstance(x, tuple) else x).shape))

        if t["io"] == "in":
            mod.register_forward_pre_hook(lambda m, a, key=key, i=index: record(key, a, i))
        else:
            mod.register_forward_hook(lambda m, a, o, key=key, i=index: record(key, o, i))
    with torch.no_grad():
        model(*dummy_batch(cfg, sym))
    for step in steps:
        t = step["trace"]
        actual = (tuple(model.get_parameter(t["param"]).shape) if "param" in t
                  else seen.get((t["module"], t["io"], t.get("index", 0))))
        expected = tuple(shape_numbers(step["shape"], cfg, step["symbols"], B))
        assert actual == expected, f"{preset}: {step['name']} ({t})"


def test_observation_groups_sum_to_d_o():
    cfg, choices = compose_preset("E001_baseline")
    specs = load_specs()
    assert sum(specs["obs_groups"][g] for g in cfg["env"]["obs_groups"]) == symbol_values(cfg, choices)["D_o"] == 192


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_export_round_trip_through_hydra(tmp_path):
    """Files exported by the page (tests/js/export_cases.js) compose with Hydra to exactly the edited config."""
    configs = tmp_path / "configs"
    shutil.copytree(CONFIGS, configs)
    (tmp_path / "sources.json").write_text(json.dumps(config_sources(configs)))
    out = tmp_path / "out"
    out.mkdir()
    subprocess.run([NODE, str(REPO / "tests" / "js" / "export_cases.js"), str(tmp_path / "sources.json"), str(out)],
                   check=True)
    expected = json.loads((out / "expected.json").read_text())
    for name, exp in expected.items():
        shutil.copy(out / f"{name}.yaml", configs / "experiment" / f"{name}.yaml")
        cfg, choices = compose_preset(name, configs)
        assert {g: choices[g] for g in exp["choices"]} == exp["choices"], name
        assert cfg == exp["cfg"], name
        for path, value in exp["edits"].items():
            node = cfg
            for key in path.split("."):
                node = node[key]
            assert node == value, f"{name}: {path}"
        assert (cfg["name"], cfg["description"]) == (exp["name"], exp["description"])
        args = shlex.split(exp["command"])[2:]  # the command-line form gives the same config
        assert compose_preset(None, configs, args)[0] == cfg, name


def write_run(root: Path, name: str, stages: list[tuple[str, int]], status: str, partial: bool = False) -> Path:
    run = root / f"20261201-1000{len(list(root.glob('*'))):02d}_{name}"
    (run / "videos").mkdir(parents=True)
    rows, step = [], 0
    for stage, iters in stages:
        for it in range(iters):
            step += 1024
            rows.append({"step": step, "iter": it, "stage": stage, "time": it * 0.5,
                         f"loss/{stage}": 1.0 / (it + 1), "perf/env_steps_per_s": 5e4 + it})
        (run / "videos" / f"{stage}_{0:05d}.mp4").write_bytes(b"old")
        (run / "videos" / f"{stage}_{iters - 1:05d}.mp4").write_bytes(b"new")
    text = "\n".join(json.dumps(r) for r in rows) + ("\n{\"step\": 99" if partial else "\n")
    (run / "metrics.jsonl").write_text(text)
    (run / "summary.json").write_text(json.dumps({
        "status": status, "name": name, "description": f"fake {name}", "gpu_name": "TEST GPU", "n_envs": 64,
        "env_steps": step, "choices": {"matcher": "latent", "train": "staged", "experiment": None},
        "overrides": ["matcher=latent", "history=10", "init_from=/home/someone/runs/x"]}))
    return run


def test_build_with_fake_runs(tmp_path):
    runs = tmp_path / "runs"
    done = write_run(runs, "E003_matcher_latent", [("contrastive", 400), ("confidence", 50), ("policy_ppo", 20)], "done")
    write_run(runs, "live", [("contrastive", 7)], "running", partial=True)
    (runs / "dev").mkdir()
    (runs / "dev" / "table_sanity.mp4").write_bytes(b"video")
    notes = tmp_path / "notes.yaml"
    notes.write_text("E003_matcher_latent: Learns slower than the baseline.\n")
    page, data, copied = build(runs, tmp_path / "build", notes)
    first, live = data["runs"]
    assert [r["status"] for r in data["runs"]] == ["done", "running"]
    assert first["note"] == "Learns slower than the baseline." and first["gpu"] == "TEST GPU"
    assert len(first["curves"]["step"]) <= 260 and first["curves"]["step"][-1] == 470 * 1024
    assert {"loss/contrastive", "loss/confidence", "perf/env_steps_per_s"} <= set(first["curves"]["metrics"])
    assert sorted(v["file"] for v in first["videos"]) == ["confidence_00049.mp4", "contrastive_00399.mp4",
                                                          "policy_ppo_00019.mp4"]
    assert first["chips"] == ["matcher=latent", "history=10", "init_from=.../x"]
    assert first["arch"] == {"choices": {"matcher": "latent", "train": "staged"},
                             "values": {"history": 10}}
    assert len(live["curves"]["step"]) == 7  # the partly written line is skipped
    assert "E003_matcher_latent" not in {p["id"] for p in data["planned"]}
    assert (tmp_path / "build" / f"videos/{done.name}/contrastive_00399.mp4").read_bytes() == b"new"
    assert "videos/dev/table_sanity.mp4" in copied
    html = page.read_text()
    assert html.startswith("<title>Tactile Object Classification</title>") and "{{" not in html.split("<script")[0]
    assert "/home/someone" not in html and str(tmp_path) not in html


def test_empty_state_has_no_numbers(tmp_path):
    (tmp_path / "runs").mkdir()
    page, data, copied = build(tmp_path / "runs", tmp_path / "build", tmp_path / "missing.yaml")
    assert data["runs"] == [] and data["envCheck"] is None and copied == {}
    assert len(data["planned"]) == len(PRESETS) and data["planned"][0]["status"] == "planned"
    assert "No training runs yet." in page.read_text()


def test_downsample_keeps_stage_edges():
    rows = [{"step": i, "stage": "a" if i < 1000 else "b", "loss/x": float(i)} for i in range(2000)]
    d = downsample(rows, 100)
    assert len(d["step"]) < 120 and {999, 1000, 1999} <= set(d["step"])


def test_parse_overrides():
    assert parse_overrides(["+experiment=E007_history_10", "matcher=latent", "train.stages.joint.lr=1e-4",
                            "hydra.verbose=true"], {"matcher"}) == ("E007_history_10", {"matcher": "latent"},
                                                                    {"train.stages.joint.lr": 1e-4})
