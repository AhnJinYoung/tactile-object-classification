"""Launcher tests: sweep expansion and the GPU scheduling with fake GPUs and a dummy command (no GPU needed)."""

import importlib.util
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("launch", REPO / "scripts" / "launch.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)

DUMMY = """import os, sys, time
open(sys.argv[1], "a").write(f"{os.environ['CUDA_VISIBLE_DEVICES']} {time.time()} start\\n")
time.sleep(0.4)
open(sys.argv[1], "a").write(f"{os.environ['CUDA_VISIBLE_DEVICES']} {time.time()} end\\n")
sys.exit(3 if "fail" in sys.argv[2] else 0)
"""


def test_expand_sweep():
    spec = {"common": ["n_envs=8"], "runs": [["+experiment=A"], "+experiment=B seed=1"], "grid": {"lr": [1, 2]}}
    assert launch.expand_sweep(spec) == [["n_envs=8", "+experiment=A", "lr=1"], ["n_envs=8", "+experiment=A", "lr=2"],
                                         ["n_envs=8", "+experiment=B", "seed=1", "lr=1"],
                                         ["n_envs=8", "+experiment=B", "seed=1", "lr=2"]]
    first = yaml.safe_load((REPO / "configs" / "sweep" / "first_batch.yaml").read_text())
    runs = launch.expand_sweep(first)
    assert len(runs) == 13 and all((REPO / "configs" / "experiment" / f"{r[0].split('=')[1]}.yaml").is_file()
                                   for r in runs)


def test_schedule_fake_gpus(tmp_path):
    """5 jobs on GPUs [3, 7] with 2 slots each: never more than 2 at once per GPU, a failure does not stop the rest."""
    trace = tmp_path / "trace.txt"
    (tmp_path / "dummy.py").write_text(DUMMY)
    names = ["a", "b", "fail", "c", "d"]
    jobs = [launch.Job(k, [f"name={n}"], tmp_path / f"{k}.log") for k, n in enumerate(names)]
    t0 = time.time()
    done = launch.schedule(jobs, [3, 7], 2, lambda j: [sys.executable, str(tmp_path / "dummy.py"), str(trace),
                                                         j.overrides[0]], poll=0.05)
    assert [j.returncode for j in done] == [0, 0, 3, 0, 0]
    assert {j.gpu for j in done} == {3, 7} and time.time() - t0 < 10
    events = sorted((float(t), gpu, kind) for gpu, t, kind in (line.split() for line in trace.read_text().splitlines()))
    running = {"3": 0, "7": 0}
    for _, gpu, kind in events:
        running[gpu] += 1 if kind == "start" else -1
        assert running[gpu] <= 2
    assert all(j.log.is_file() for j in done)


def test_schedule_waits_for_free_memory(tmp_path):
    """A GPU below the free-memory floor gets no run; all runs go to the other one, one at a time."""
    (tmp_path / "dummy.py").write_text(DUMMY)
    trace = tmp_path / "trace.txt"
    jobs = [launch.Job(k, [f"name=r{k}"], tmp_path / f"{k}.log") for k in range(3)]
    done = launch.schedule(jobs, [0, 1], 1, lambda j: [sys.executable, str(tmp_path / "dummy.py"), str(trace),
                                                         j.overrides[0]],
                           free_gb=lambda g: 40.0 if g == 1 else 2.0, min_free_gb=10.0, poll=0.05)
    assert all(j.gpu == 1 and j.returncode == 0 for j in done)


def test_dry_run(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["launch.py", "first_batch", "--gpus", "0,1", "--dry-run"])
    launch.main()
    out = capsys.readouterr().out
    assert "13 runs on GPUs [0, 1]" in out and "+experiment=E013_attention_16x32" in out


def test_preflight_cpu():
    """scripts/preflight.py on the CPU: every check passes or is skipped; a requested but missing wandb fails."""
    import importlib.util as iu
    import subprocess

    if not (REPO / "third_party" / "sharpa").is_dir():
        return
    out = subprocess.run([sys.executable, str(REPO / "scripts" / "preflight.py"), "--device", "cpu",
                          "wandb.enabled=true"], capture_output=True, text=True, timeout=600)
    lines = {ln.split()[1]: ln.split()[0] for ln in out.stdout.splitlines() if ln[:4] in ("PASS", "SKIP", "WARN", "FAIL")}
    assert {lines[k] for k in ("python_torch", "assets", "runs_dir", "environment", "egl_render")} == {"PASS"}, out
    has_wandb = iu.find_spec("wandb") is not None
    assert lines["wandb"] == ("PASS" if has_wandb else "FAIL") and out.returncode == (0 if has_wandb else 1)
