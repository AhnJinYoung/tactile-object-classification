"""Live browser viewer: serves one world of a batched `BinEnv` through MuJoCo Studio and returns the user's commands.

Sits beside whatever loop steps the env (`scripts/view.py` now, the trainer later) and owns no stepping itself. Each
`sync` copies the shown world from `env.world_state(w)` into a CPU `MjData` of the same compiled model, runs
`mj_forward` and streams it; the browser renders from the model bundle the server sends once. Browser input comes back
as `Commands` (world, driver, pause / speed / single steps, resets); the viewer's own drivers (`manual`, `hold`) and
mouse perturbation are applied to the env by `drive` and `sync`. The server binds all interfaces without
authentication: reach it through an SSH tunnel (or a firewall) only.
"""

import dataclasses
import time
from collections.abc import Mapping, Sequence

import mujoco
import numpy as np
import torch
from mujoco.studio import app as studio_app
from mujoco.studio import launch_web, messages, plugin, sim, viewer_protocol

from tacfind.viz.panel import Panel, ResetWorld, SelectDriver, SelectWorld, WorldSnapshot

BUILTIN = ("manual", "hold")  # drivers provided by the viewer; batched ones (explorer, later policy) by the caller
CTRL, XFRC = int(mujoco.mjtState.mjSTATE_CTRL), int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)


@dataclasses.dataclass
class Commands:
    """What the browser user asked for since the last `sync`.

    Attributes:
        world: shown world.
        driver: who drives it: a caller's batched driver name or one of `BUILTIN`.
        paused: the viewer's pause button.
        speed: requested pace, percent of real time.
        steps: single control steps requested while paused.
        resets: worlds to reset (the caller owns resets, a trainer may refuse them).
    """

    world: int = 0
    driver: str = ""
    paused: bool = False
    speed: float = 100.0
    steps: int = 0
    resets: list[int] = dataclasses.field(default_factory=list)


class Bridge(plugin.SimPlugin):
    """Sim side: turns browser messages into `Commands`, a manual joint target and a perturbation wrench.

    Args:
        model: the compiled scene (decodes state arrays).
        n_worlds: number of worlds that can be selected.
        drivers: selectable driver names; the first is the default.
    """

    def __init__(self, model: mujoco.MjModel, n_worlds: int, drivers: Sequence[str]):
        self.model, self.n_worlds, self.drivers = model, n_worlds, tuple(drivers)
        self.cmd = Commands(driver=self.drivers[0])
        self.ctrl_edit: np.ndarray | None = None  # (nu,) joint targets from the Controls sliders, rad
        self.xfrc = np.zeros((model.nbody, 6))  # mouse perturbation [force N, torque N m] per body, base frame

    def take(self) -> Commands:
        """Current commands; the one-shot parts (steps, resets) are handed out once."""
        out = dataclasses.replace(self.cmd, resets=list(self.cmd.resets))
        self.cmd.steps, self.cmd.resets = 0, []
        return out

    @messages.handler
    def on_step_control(self, s: messages.StepControlSnapshot) -> bool:
        self.cmd.paused, self.cmd.speed = s.pause_state != sim.PauseState.UNPAUSED, float(s.speed)
        return True

    @messages.handler
    def on_single_step(self, _: messages.SingleStepEvent) -> bool:
        self.cmd.steps += 1
        return True

    @messages.handler
    def on_reset(self, _: messages.ResetEvent) -> bool:
        self.cmd.resets.append(self.cmd.world)  # the viewer's own Reset button resets the shown world
        return True

    @messages.handler
    def on_reset_world(self, _: ResetWorld) -> bool:
        self.cmd.resets.append(self.cmd.world)
        return True

    @messages.handler
    def on_select_world(self, e: SelectWorld) -> bool:
        if 0 <= e.world < self.n_worlds:
            self.cmd.world = e.world
        return True

    @messages.handler
    def on_select_driver(self, e: SelectDriver) -> bool:
        if e.driver in self.drivers:
            self.cmd.driver = e.driver
        return True

    @messages.handler
    def on_state(self, e: messages.StateEvent) -> bool:
        """A Controls slider edit switches the world to manual; joint / pose edits are ignored (state comes from the
        env, and a single world's qpos cannot be set through its API)."""
        if e.state_sig & CTRL and len(e.state) == mujoco.mj_stateSize(self.model, e.state_sig):
            d = mujoco.MjData(self.model)
            mujoco.mj_setState(self.model, d, e.state, e.state_sig)
            self.ctrl_edit, self.cmd.driver = d.ctrl.copy(), "manual"
        return True

    @messages.handler
    def on_perturb(self, s: messages.PerturbSnapshot) -> bool:
        if s.state_sig == XFRC and s.state.size == self.xfrc.size:
            self.xfrc = s.state.reshape(self.xfrc.shape).copy()
        return True

    @messages.handler(priority=messages.Priority.CRITICAL)
    def on_load_model(self, e: messages.LoadModelEvent) -> bool:
        print(f"[viewer] ignored a model load / reload request from the browser ({e.path or 'reload'})", flush=True)
        return True

    @messages.handler(priority=messages.Priority.CRITICAL)
    def on_option(self, _: messages.MjOptionSnapshot) -> bool:
        return True  # physics options belong to the env config, not to the browser

    @messages.handler(priority=messages.Priority.CRITICAL)
    def on_keyframe(self, _: messages.KeyframeEvent) -> bool:
        return True


class LiveViewer:
    """Serves world `w` of a `BinEnv` to MuJoCo Studio in the browser (one port: page, model bundle, state stream).

    Args:
        env: the environment; its public API plus `q_target`, `step_size`, `clouds` and `sim.contacts()` are used.
        drivers: names of the caller's batched drivers, the first being the default of every world; `BUILTIN` are
            appended. A trained policy plugs in as one more name here.
        port: HTTP port (0: first free port from 8080).
        title: browser tab title.
    """

    def __init__(self, env, drivers: Sequence[str] = ("explorer",), port: int = 8080, title: str = "tacfind live"):
        self.env, self.port = env, port
        self.drivers = (*drivers, *BUILTIN)
        self.model, self.data = env.mjm, mujoco.MjData(env.mjm)
        self.bridge = Bridge(self.model, env.n, self.drivers)
        self.panel = Panel(env.n, self.drivers)
        self.config = viewer_protocol.ViewerConfig(title=title, http_port=port)
        self.handle = None
        self.world, self.driver = 0, self.drivers[0]  # what `drive` last applied
        self.q_manual = None  # (28,) manual joint targets of the shown world, rad, env device
        self.wrench = None  # perturbation last sent to the env, (nbody, 6) or None
        self._key = None  # (world, driver, readings) last shown

    # ------------------------------------------------------------------------------------------------ lifecycle --
    def start(self) -> "LiveViewer":
        """Start the web server (daemon thread) and publish the model; returns self (usable as a context manager)."""
        self.handle = launch_web.launch(self.config, plugins=[studio_app.App(), self.panel, self.bridge])
        self.handle.sync(self.model, self.data)
        t0 = time.monotonic()
        while self.panel._viewer is None and time.monotonic() - t0 < 10:  # the viewer thread binds the port
            time.sleep(0.01)
        self.port = self.panel.viewer.port
        return self

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def __enter__(self) -> "LiveViewer":
        return self.start()

    def __exit__(self, *_) -> None:
        self.close()

    def running(self) -> bool:
        return self.handle is not None and self.handle.is_running()

    # ------------------------------------------------------------------------------------------------ per frame --
    def sync(self) -> Commands:
        """Show the current state of the selected world, exchange messages with the browser and forward the mouse
        perturbation to the env. Call at display rate (~60 Hz), whether or not the env stepped."""
        self._show()
        self.handle.sync(self.model, self.data)
        cmd = self.bridge.take()
        self._perturb(cmd.world)
        self._select(cmd)
        return cmd

    def drive(self, actions: Mapping[str, torch.Tensor] | None = None) -> None:
        """Before `env.step`: drive the shown world as the user chose, through `env.override`.

        Args:
            actions: batched actions (N, 28) by driver name, for a selected driver other than the default.

        Manual mapping: the Controls sliders set absolute joint targets q_des (28, rad, actuator order); each control
        step sends action = clip((q_des - q_target) / step_size, -1, 1), i.e. the target moves toward the slider at
        the env's own rate limit (arm 0.6 rad/s, hand 0.83 rad/s) and stays inside the env's joint and lead limits.
        `hold` sends zeros (targets frozen); the default driver releases the override.
        """
        env = self.env
        if self.driver == "manual":
            env.override(self.world, ((self.q_manual - env.q_target[self.world]) / env.step_size).clamp(-1, 1))
        elif self.driver == "hold":
            env.override(self.world, torch.zeros_like(env.q_target[self.world]))
        elif self.driver != self.drivers[0] and actions is not None and self.driver in actions:
            env.override(self.world, actions[self.driver][self.world])
        else:
            env.override(self.world, None)

    # ------------------------------------------------------------------------------------------------ internals --
    def _select(self, cmd: Commands) -> None:
        """Apply a world / driver change and a slider edit (the old world goes back to the default driver)."""
        env = self.env
        if cmd.world != self.world:
            env.override(self.world, None)
        if (cmd.world, cmd.driver) != (self.world, self.driver):
            self.q_manual = env.q_target[cmd.world].clone()  # entering manual holds the current targets
        self.world, self.driver = cmd.world, cmd.driver
        if self.bridge.ctrl_edit is not None:
            self.q_manual = torch.as_tensor(self.bridge.ctrl_edit, dtype=torch.float32, device=env.device)
            self.bridge.ctrl_edit = None
            self._key = None  # show the new slider positions at once

    def _perturb(self, world: int) -> None:
        """Forward the browser's wrench (only on change); a world switch clears it on the old world."""
        if world != self.world and self.wrench is not None:
            self._send_perturb(self.world, None)
        wrench = self.bridge.xfrc if self.bridge.xfrc.any() else None
        if wrench is not None or self.wrench is not None:
            if wrench is None or self.wrench is None or not np.array_equal(wrench, self.wrench):
                self._send_perturb(world, wrench)

    def _send_perturb(self, world: int, wrench: np.ndarray | None) -> None:
        self.wrench = wrench
        self.env.perturb(world, None if wrench is None else torch.as_tensor(wrench, dtype=torch.float32))

    def _show(self) -> None:
        """Refresh the view data and send the panel snapshot when the env has new readings or the world changed."""
        env, w = self.env, self.world
        key = (w, self.driver, id(env.reading))  # `env.reading` is replaced on every step and reset
        if key == self._key:
            return
        self._key = key
        s = env.world_state(w)
        d = self.data
        d.qpos[:], d.qvel[:] = s["qpos"], 0.0
        d.ctrl[:] = (self.q_manual if self.driver == "manual" and self.q_manual is not None
                     else env.q_target[w]).cpu().numpy()  # slider positions: what drives the world
        d.time = float(s["t"]) * env.cfg.timestep * env.cfg.substeps
        mujoco.mj_forward(self.model, d)
        c = {k: v.cpu().numpy() for k, v in env.sim.contacts().items()}
        sel = c["valid"] & (c["world"] == w)
        tgt_body = int(s["active_body"][s["target_slot"]])
        radius = env.clouds[int(s["target_lib"]), :, :3].norm(dim=-1).max().item()
        touched = int(s.get("touched", -1))
        lines = (f"t {d.time:5.2f} s   step {int(s['t'])}",
                 f"target  object {int(s['target_lib'])} (slot {int(s['target_slot'])})",
                 f"touched {'object ' + str(touched) if touched >= 0 else '-'}"
                 f"   on target {bool(s.get('touch_target', False))}",
                 f"reward {float(s.get('reward', 0.0)):+.3f}",
                 f"pushing {'yes' if self.wrench is not None else 'no'}")
        self.handle.send_to_viewer(WorldSnapshot(
            world=w, driver=self.driver, lines=lines, conf=float(s.get("conf_gt", 0.0)),
            tip_force=s["force"] * s["pts_valid"], tip_pos=s["pts"], tip_normal=s["normals"],
            contact_pos=c["pos"][sel], contact_normal=c["frame"][sel, 0], contact_force=c["force"][sel, 0],
            target_pos=d.xpos[tgt_body].copy(), target_radius=radius + 0.01))
