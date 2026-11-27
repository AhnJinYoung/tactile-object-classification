"""Watch and drive one simulated bin world live in the browser (MuJoCo Studio), stepped at real time.

    python scripts/view.py                              # GPU server: MuJoCo Warp, 4 worlds, port 8080
    python scripts/view.py sim=cpu --worlds 2           # laptop / no GPU: CPU reference backend
    python scripts/view.py --port 8090 label=binary n_slots=3   # key=value: overrides of configs/env/bin.yaml

Then open the printed URL; from a laptop, first open the printed SSH tunnel (-C compresses the one-time model
download, about 61 MB raw / 27 MB compressed, mostly the hand and arm meshes). The page offers: world choice, reset,
driver (explorer | manual | hold), the viewer's own pause / step / speed buttons, Controls sliders (editing one
switches the world to manual), Ctrl + mouse drag to push bodies, read-outs and contact overlays.

SECURITY: the viewer listens on ALL network interfaces with NO authentication or encryption; anyone who can reach
the port can watch and control the simulation. Use it only behind an SSH tunnel or a firewall that blocks the port.
"""

import argparse
import getpass
import socket
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from tacfind.envs import BinEnv
from tacfind.envs.explorer import Explorer
from tacfind.viz import LiveViewer

REPO = Path(__file__).resolve().parents[1]
FRAME = 1 / 60  # display period, s (the state stream runs at about 60 Hz)


def reset_world(env: BinEnv, obs: dict, w: int) -> None:
    """Start a new episode in world w only."""
    mask = torch.zeros(env.n, dtype=torch.bool, device=env.device)
    mask[w] = True
    env.reset_worlds(mask)
    obs["first"][w] = True  # restarts the explorer's state machine for that world


def run(env: BinEnv, viewer: LiveViewer, drivers: dict, seconds: float = 0.0) -> None:
    """Real-time loop: sync the browser at ~60 Hz, step the env at 20 Hz x the requested speed, until the viewer
    closes or `seconds` (0 = forever) have passed.

    Args:
        drivers: batched drivers by name, `obs -> (N, 28)` actions; the first drives every world not overridden.
    """
    dt = env.cfg.timestep * env.cfg.substeps
    default = next(iter(drivers))
    obs = env.reset()
    budget, t0 = 0.0, time.monotonic()
    last = report = t0
    n_steps = n_frames = 0
    while viewer.running() and not (seconds and time.monotonic() - t0 > seconds):
        tick = time.monotonic()
        cmd = viewer.sync()
        for w in dict.fromkeys(cmd.resets):
            reset_world(env, obs, w)
        if cmd.paused:
            budget, n = 0.0, cmd.steps
        else:  # sim time owed; capped so a slow step never builds a backlog
            budget = min(budget + (tick - last) * cmd.speed / 100, 2 * dt)
            n = int(budget // dt)
            budget -= n * dt
        last = tick
        for _ in range(n):
            actions = {name: fn(obs) for name, fn in drivers.items() if name in (default, cmd.driver)}
            viewer.drive(actions)
            obs, _, _, _ = env.step(actions[default])
            n_steps += 1
        n_frames += 1
        if tick - report > 5:
            print(f"[view] world {cmd.world} driver {cmd.driver} paused {cmd.paused}: {n_frames / (tick - report):.0f}"
                  f" frames/s, real-time factor {n_steps * dt / (tick - report):.2f}", flush=True)
            report, n_steps, n_frames = tick, 0, 0
        time.sleep(max(0.0, FRAME - (time.monotonic() - tick)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("overrides", nargs="*", help="key=value overrides of the env config (configs/env/bin.yaml)")
    ap.add_argument("--worlds", type=int, default=4, help="number of simulated worlds")
    ap.add_argument("--port", type=int, default=8080, help="HTTP port (0: first free port from 8080)")
    ap.add_argument("--device", default="auto", help="simulation device: auto | cuda | cuda:1 | cpu")
    ap.add_argument("--bank", type=int, default=64, help="settled start states built at start-up (env bank_size)")
    ap.add_argument("--history", type=int, default=30, help="H, steps of contact memory behind the coverage label")
    ap.add_argument("--split", default="train", choices=["train", "heldout"], help="object library split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=0.0, help="stop after this many seconds (0: run until Ctrl+C)")
    args = ap.parse_args()

    cfg = OmegaConf.load(REPO / "configs" / "env" / "bin.yaml")
    cfg.bank_size = args.bank
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([o.removeprefix("env.") for o in args.overrides]))
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if cfg.sim == "warp" and device == "cpu":
        print("[view] MuJoCo Warp on the CPU is far slower than real time; pass sim=cpu without a GPU", flush=True)
    print(f"[view] building {args.worlds} worlds ({cfg.sim} on {device}, bank {cfg.bank_size}) ...", flush=True)
    env = BinEnv(cfg, args.worlds, args.history, device, args.seed, args.split)
    drivers = {"explorer": Explorer(env, args.seed).act}  # a trained policy plugs in here as another name
    with LiveViewer(env, tuple(drivers), args.port) as viewer:
        port = viewer.port
        print(f"\n[view] open http://localhost:{port}/ in a browser. From another machine, first run there:\n"
              f"         ssh -N -C -L {port}:localhost:{port} {getpass.getuser()}@{socket.getfqdn()}\n"
              f"       The port is open on all interfaces without authentication: keep it firewalled.\n", flush=True)
        try:
            run(env, viewer, drivers, args.seconds)
        except KeyboardInterrupt:
            print("\n[view] stopped", flush=True)


if __name__ == "__main__":
    main()
