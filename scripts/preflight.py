"""Check a new machine before any run: one PASS / WARN / SKIP / FAIL line per item, exit code 1 on any FAIL.

    python scripts/preflight.py                         # GPU server: cuda, MuJoCo Warp
    python scripts/preflight.py --device cpu            # no GPU: CPU reference backend (sim=cpu)
    python scripts/preflight.py wandb.enabled=true      # trailing key=value: overrides of configs/config.yaml
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")  # headless rendering; must precede the mujoco import

import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


class Skip(Exception):
    """Item does not apply here."""


class Warn(Exception):
    """Item works but needs attention."""


def checks(cfg, device: str, state: dict) -> dict:
    """Name -> check; each returns a message (PASS) or raises (Skip / Warn / anything else = FAIL)."""
    cuda = device.startswith("cuda")

    def python_torch():
        assert torch.cuda.is_available() or not cuda, "CUDA requested but torch sees no GPU"
        return f"python {sys.version.split()[0]}, torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}"

    def gpus():
        if not torch.cuda.is_available():
            raise Skip("no GPU")
        return "; ".join(f"{i}: {torch.cuda.get_device_name(i)}, {torch.cuda.mem_get_info(i)[0] / 2**30:.1f} GB free"
                         for i in range(torch.cuda.device_count()))

    def driver():
        if not cuda:
            raise Skip("cpu run")
        m = re.search(r"CUDA Version: (\d+)\.(\d+)", subprocess.run(["nvidia-smi"], capture_output=True,
                                                                     text=True).stdout)
        assert m, "nvidia-smi reports no CUDA version"
        version = (int(m[1]), int(m[2]))
        if version < (12, 4) and cfg.env.get("solver_iterations") is None:
            raise Warn(f"driver CUDA {m[1]}.{m[2]} < 12.4: set env.solver_iterations (configs/env/bin.yaml)")
        return f"driver CUDA {m[1]}.{m[2]}"

    def warp_device():
        import warp as wp
        wp.config.quiet = True
        wp.init()
        return f"warp {wp.__version__}, device {wp.get_device('cuda:0' if cuda else 'cpu')}"

    def assets():
        missing = [p for p in ("sharpa", "mujoco_menagerie") if not (REPO / cfg.env.assets / p).is_dir()]
        assert not missing, f"missing {missing}: run scripts/fetch_assets.py"
        return str(REPO / cfg.env.assets)

    def runs_dir():
        root = REPO / cfg.runs_dir
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=root):
            pass
        free = shutil.disk_usage(root).free / 2**30
        assert free > 5, f"only {free:.1f} GB free under {root}"
        return f"{root} writable, {free:.0f} GB free"

    def environment():
        from tacfind.envs.bin_env import BinEnv
        from tacfind.envs.explorer import Explorer
        env = state["env"] = BinEnv(cfg.env, 2, cfg.history, device, cfg.seed)
        explorer, obs = Explorer(env), env.reset()
        for _ in range(5):
            obs, reward, _, _ = env.step(explorer.act(obs))
        assert torch.isfinite(obs["o"]).all() and torch.isfinite(reward).all(), "non-finite observation or reward"
        return f"{cfg.env.sim} backend on {env.device}: 2 worlds stepped 5 times, obs_dim {env.spec.obs_dim}"

    def egl_render():
        from tacfind.envs.render import WorldRenderer
        assert "env" in state, "needs the environment"
        renderer = WorldRenderer(state["env"])
        frame = renderer(0)
        renderer.close()
        assert frame.std() > 0, "blank frame"
        return f"MUJOCO_GL={os.environ['MUJOCO_GL']}, frame {frame.shape}"

    def wandb():
        if not cfg.wandb.enabled:
            raise Skip("wandb.enabled is false")
        import wandb as wb
        return f"wandb {wb.__version__} (log in with WANDB_API_KEY or `wandb login`)"

    return dict(python_torch=python_torch, gpus=gpus, cuda_driver=driver, warp_device=warp_device, assets=assets,
                runs_dir=runs_dir, environment=environment, egl_render=egl_render, wandb=wandb)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | cpu (cpu also selects sim=cpu)")
    ap.add_argument("overrides", nargs="*", help="config overrides key=value")
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    extra = ["env.sim=cpu"] if device == "cpu" else []
    with initialize_config_dir(str(REPO / "configs"), version_base="1.3"):
        cfg = compose("config", [*extra, "env.bank_size=4", *args.overrides])
    failed, state = 0, {}
    for name, fn in checks(cfg, device, state).items():
        try:
            status, msg = "PASS", fn()
        except Skip as e:
            status, msg = "SKIP", str(e)
        except Warn as e:
            status, msg = "WARN", str(e)
        except Exception as e:  # noqa: BLE001 - every failure is reported, the other checks still run
            status, msg, failed = "FAIL", f"{type(e).__name__}: {e}", failed + 1
        print(f"{status}  {name:12s} {msg}", flush=True)
    if "env" in state and hasattr(state["env"].sim, "pool"):
        state["env"].sim.pool.shutdown()
    print(f"{'all checks passed' if not failed else f'{failed} check(s) FAILED'} (device {device})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
