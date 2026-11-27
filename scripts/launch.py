"""Run a list of experiments in parallel over the GPUs of this machine, one run per GPU slot (CUDA_VISIBLE_DEVICES).

    python scripts/launch.py first_batch                     # configs/sweep/first_batch.yaml on every GPU
    python scripts/launch.py first_batch --gpus 0,2 --per-gpu 2 --min-free-gb 20 --dry-run

The sweep file lists runs as override strings; an optional `grid` crosses every run with every combination, and
`common` is added to every run. Queued runs start when a slot frees up and its GPU has the requested free memory.
Each run's console output goes to runs/_launch/<timestamp>_<sweep>/<k>_<label>.log; failures are reported at the end
without stopping the other runs; Ctrl-C stops all children.
"""

import argparse
import itertools
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]


def expand_sweep(spec: dict) -> list[list[str]]:
    """Sweep YAML content -> one override list per run (runs x grid combinations, plus `common`)."""
    runs = [list(r) if isinstance(r, list) else str(r).split() for r in spec.get("runs") or [[]]]
    grid = spec.get("grid") or {}
    combos = [[f"{k}={v}" for k, v in zip(grid, vals)] for vals in itertools.product(*grid.values())] or [[]]
    return [list(spec.get("common") or []) + r + c for r in runs for c in combos]


def list_gpus() -> list[dict]:
    """GPUs from nvidia-smi: [{"index", "name", "free_gb"}] (empty when nvidia-smi is missing)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = [[c.strip() for c in line.split(",")] for line in out.strip().splitlines()]
    return [{"index": int(i), "name": name, "free_gb": float(free) / 1024} for i, name, free in rows]


def label(overrides: list[str]) -> str:
    """Short file-name label of a run from its overrides."""
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", "_".join(o.split("=")[-1] for o in overrides))[:80] or "default"


@dataclass
class Job:
    """One queued run."""

    k: int
    overrides: list[str]
    log: Path
    gpu: int | None = None
    proc: subprocess.Popen | None = None
    returncode: int | None = None
    log_file: object = field(default=None, repr=False)


def schedule(jobs: list[Job], gpus: list[int], per_gpu: int, command: Callable[[Job], list[str]],
             free_gb: Callable[[int], float] = lambda g: float("inf"), min_free_gb: float = 0.0,
             poll: float = 2.0) -> list[Job]:
    """Run jobs with at most `per_gpu` at a time on each GPU; returns the jobs with their return codes.

    Args:
        jobs: queue in order. gpus: GPU indices to use. per_gpu: concurrent runs per GPU.
        command: job -> argv. free_gb: GPU index -> free memory now, GB. min_free_gb: start only above this.
        poll: seconds between checks.
    """
    queue, running = list(jobs), []
    try:
        while queue or running:
            for job in [j for j in running if j.proc.poll() is not None]:
                job.returncode = job.proc.returncode
                job.log_file.close()
                running.remove(job)
                print(f"[{'ok' if job.returncode == 0 else 'FAILED'}] #{job.k} gpu {job.gpu}: "
                      f"{' '.join(job.overrides)} (exit {job.returncode})", flush=True)
            for gpu in gpus:
                while queue and sum(j.gpu == gpu for j in running) < per_gpu and free_gb(gpu) >= min_free_gb:
                    job = queue.pop(0)
                    job.gpu, job.log_file = gpu, open(job.log, "w")
                    env = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu)}
                    job.proc = subprocess.Popen(command(job), stdout=job.log_file, stderr=subprocess.STDOUT, env=env,
                                                cwd=REPO, start_new_session=True)  # own group: Ctrl-C handled here
                    running.append(job)
                    print(f"[start] #{job.k} gpu {gpu}: {' '.join(job.overrides)} -> {job.log}", flush=True)
                    break  # re-check free memory after the run has claimed some
            time.sleep(poll)
    except KeyboardInterrupt:
        print("stopping all runs ...", flush=True)
        for job in running:
            os.killpg(job.proc.pid, signal.SIGINT)
        deadline = time.time() + 30
        for job in running:
            try:
                job.proc.wait(max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                os.killpg(job.proc.pid, signal.SIGKILL)
                job.proc.wait()
            job.returncode = job.proc.returncode
        raise
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", help="name in configs/sweep/ or a path to a sweep YAML")
    ap.add_argument("--gpus", help="comma-separated GPU indices (default: all from nvidia-smi)")
    ap.add_argument("--per-gpu", type=int, default=1, help="concurrent runs per GPU")
    ap.add_argument("--min-free-gb", type=float, default=0.0, help="start a run only when its GPU has this free")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()
    path = Path(args.sweep) if args.sweep.endswith(".yaml") else REPO / "configs" / "sweep" / f"{args.sweep}.yaml"
    runs = expand_sweep(yaml.safe_load(path.read_text()))
    found = list_gpus()
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else [g["index"] for g in found]
    log_dir = REPO / "runs" / "_launch" / f"{datetime.now():%Y%m%d-%H%M%S}_{path.stem}"
    jobs = [Job(k, o, log_dir / f"{k:03d}_{label(o)}.log") for k, o in enumerate(runs)]
    print(f"{len(jobs)} runs on GPUs {gpus} x {args.per_gpu} "
          f"({', '.join(f'{g['index']}: {g['name']}' for g in found) or 'nvidia-smi found no GPU'})")
    for job in jobs:
        print(f"  #{job.k}: python scripts/train.py {' '.join(job.overrides)}")
    if args.dry_run:
        return
    if not gpus:
        sys.exit("no GPU to run on (use --gpus)")
    log_dir.mkdir(parents=True)
    free = lambda g: next((x["free_gb"] for x in list_gpus() if x["index"] == g), 0.0)
    done = schedule(jobs, gpus, args.per_gpu, lambda j: [sys.executable, "scripts/train.py", *j.overrides],
                    free if args.min_free_gb > 0 else (lambda g: float("inf")), args.min_free_gb)
    failed = [j for j in done if j.returncode != 0]
    print(f"{len(done) - len(failed)} / {len(done)} runs succeeded; logs in {log_dir}")
    for j in failed:
        print(f"  FAILED #{j.k} (exit {j.returncode}): {' '.join(j.overrides)} -> {j.log}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
