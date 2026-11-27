"""Run directory and logging: resolved config, git state, metrics.jsonl, summary.json, TensorBoard, optional W&B.

Every run writes only to its own `runs/<YYYYMMDD-HHMMSS>_<name>/`, so parallel runs never share a file; tables over
runs are built by scanning `runs/*/summary.json` (`scan_runs`).
"""

import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

REPO = Path(__file__).resolve().parents[2]


def make_run_dir(root: Path, name: str) -> Path:
    """Create `root/<timestamp>_<name>` atomically, adding a suffix when it exists (runs start concurrently)."""
    base = f"{datetime.now():%Y%m%d-%H%M%S}_{name}"
    root.mkdir(parents=True, exist_ok=True)
    for k in range(1000):
        path = root / (base if k == 0 else f"{base}_{k}")
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"no free run directory for {base}")


def finite(v):
    """JSON-safe scalar: None / NaN / inf -> None."""
    return None if v is None or not math.isfinite(v := float(v)) else v


def git_state(run_dir: Path) -> None:
    """Write git.txt (commit, branch, dirty flag) and patch.diff when the work tree is dirty."""
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    try:
        status = git("status", "--porcelain")
        (run_dir / "git.txt").write_text(f"commit: {git('rev-parse', 'HEAD')}\n"
                                         f"branch: {git('rev-parse', '--abbrev-ref', 'HEAD')}\ndirty: {bool(status)}\n")
        if status:
            (run_dir / "patch.diff").write_text(git("diff", "HEAD") + "\n")
    except OSError:  # no git on this machine
        (run_dir / "git.txt").write_text("commit: unknown\n")


class RunLogger:
    """Writes one run directory.

    Args:
        cfg: resolved root config (keys name, description, runs_dir, wandb).
        meta: extra summary fields (choices, overrides, kind, ...).
    """

    def __init__(self, cfg: DictConfig, meta: dict | None = None):
        self.dir = make_run_dir(REPO / cfg.runs_dir, cfg.name)
        for sub in ("videos", "figures", "ckpt"):
            (self.dir / sub).mkdir()
        OmegaConf.save(cfg, self.dir / "config.yaml", resolve=True)
        git_state(self.dir)
        self.t0, self.records = time.time(), []
        self.summary = {"status": "running", "name": cfg.name, "description": cfg.get("description", ""),
                        "start_time": datetime.now().isoformat(timespec="seconds"), "end_time": None,
                        "run_dir": self.dir.name, **(meta or {}), "stages": {}}
        from torch.utils.tensorboard import SummaryWriter
        self.tb = SummaryWriter(str(self.dir / "tb"))
        self.wandb = None
        if cfg.wandb.enabled:
            import wandb  # optional dependency: pip install wandb
            self.wandb = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=self.dir.name,
                                    dir=str(self.dir), config=OmegaConf.to_container(cfg, resolve=True))
        self.write_summary()

    def write_summary(self, **fields) -> None:
        """Update and atomically rewrite summary.json."""
        self.summary |= fields
        tmp = self.dir / "summary.json.tmp"
        tmp.write_text(json.dumps(self.summary, indent=1))
        tmp.replace(self.dir / "summary.json")

    def log(self, step: int, it: int, stage: str, scalars: dict[str, float]) -> dict:
        """Append one record to metrics.jsonl, TensorBoard and W&B; update the summary. Returns the record."""
        rec = {"step": step, "iter": it, "stage": stage, "time": round(time.time() - self.t0, 2)}
        rec |= {k: finite(v) for k, v in sorted(scalars.items())}
        with open(self.dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        self.records.append(rec)
        for k, v in rec.items():
            if "/" in k and v is not None:
                self.tb.add_scalar(k, v, step)
        if self.wandb:
            self.wandb.log({k: v for k, v in rec.items() if "/" in k} | {"iter": it}, step=step)
        self.summary["stages"][stage] = {k: v for k, v in rec.items() if k != "stage"}
        self.write_summary(env_steps=step, peak_gpu_mem_gb=gpu_mem_gb())
        return rec

    def video(self, path: Path, step: int) -> None:
        """Send a written video to W&B (TensorBoard holds only scalars)."""
        if self.wandb:
            import wandb
            self.wandb.log({"video": wandb.Video(str(path), format="mp4")}, step=step)

    def close(self, status: str) -> None:
        """Final summary (status "done" | "failed")."""
        self.write_summary(status=status, end_time=datetime.now().isoformat(timespec="seconds"),
                           peak_gpu_mem_gb=gpu_mem_gb())
        self.tb.close()
        if self.wandb:
            self.wandb.finish(exit_code=0 if status == "done" else 1)


def gpu_mem_gb() -> float | None:
    """Peak GPU memory allocated by this process, GB (None on the CPU)."""
    return torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None


def scan_runs(root: Path = REPO / "runs") -> list[dict]:
    """All run summaries under `root`, oldest first (the table of runs; there is no shared index file)."""
    return [json.loads(p.read_text()) for p in sorted(root.glob("*/summary.json"))]


def rates(s: dict) -> dict:
    """Rollout sums -> logged rates (NaN when no episode finished)."""
    n = s["episodes"]
    return {"rollout/return": s["ret"] / n, "rollout/success_rate": s["success"] / n,
            "rollout/wrong_pick_rate": s["wrong"] / n, "rollout/episode_len": s["len"] / n,
            "rollout/episodes": n, "rollout/contact_frac": s["contact"] / s["steps"],
            "rollout/touch_target_frac": s["touch_target"] / s["steps"]}


def to_floats(scalars: dict, device) -> dict[str, float]:
    """Scalar tensors on the device -> floats with one host sync."""
    names = list(scalars)
    return dict(zip(names, torch.stack([torch.as_tensor(scalars[k], device=device).float() for k in names]).tolist()))
