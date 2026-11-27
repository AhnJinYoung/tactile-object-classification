"""Browser side of the live viewer: the "Live" Dear ImGui window, the 3-D overlays and the messages they exchange.

Runs on the MuJoCo Studio viewer thread (`tacfind/viz/live.py` is the simulation side). Its handlers run only while a
controlling browser is connected; it sends requests to the simulation and draws what the latest `WorldSnapshot` says.
"""

import dataclasses
from collections.abc import Sequence

import mujoco
import numpy as np
from mujoco.studio import messages, plugin
from mujoco.studio.dear_imgui import dear_imgui as imgui

from tacfind.envs.scene import FINGERS


@dataclasses.dataclass(frozen=True)
class SelectWorld(messages.Event):
    world: int


@dataclasses.dataclass(frozen=True)
class SelectDriver(messages.Event):
    driver: str


@dataclasses.dataclass(frozen=True)
class ResetWorld(messages.Event):
    """Reset the shown world."""


@dataclasses.dataclass(frozen=True, eq=False)
class WorldSnapshot(messages.Snapshot):
    """Sim -> panel, latest wins: read-outs and overlay geometry of the shown world (base frame, m, N)."""

    world: int
    driver: str
    lines: tuple[str, ...]
    conf: float
    tip_force: np.ndarray  # (5,)
    tip_pos: np.ndarray  # (5, 3) pad contact points
    tip_normal: np.ndarray  # (5, 3)
    contact_pos: np.ndarray  # (C, 3) every contact of the world, from the simulation
    contact_normal: np.ndarray  # (C, 3) geom0 -> geom1
    contact_force: np.ndarray  # (C,) normal force
    target_pos: np.ndarray  # (3,)
    target_radius: float


def _geom(kind: mujoco.mjtGeom, size, pos, rgba) -> mujoco.MjvGeom:
    g = mujoco.MjvGeom()
    mujoco.mjv_initGeom(g, int(kind), np.asarray(size, float), np.asarray(pos, float), np.eye(3).ravel(),
                        np.asarray(rgba, np.float32))
    return g


def _arrow(p0: np.ndarray, p1: np.ndarray, width: float, rgba) -> mujoco.MjvGeom:
    g = _geom(mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), rgba)
    mujoco.mjv_connector(g, int(mujoco.mjtGeom.mjGEOM_ARROW), width, np.asarray(p0, float), np.asarray(p1, float))
    return g


def overlay_geoms(s: WorldSnapshot) -> list[mujoco.MjvGeom]:
    """Target highlight, every contact (yellow, normal arrow ~ force) and the fingertip pad contacts (red)."""
    out = [_geom(mujoco.mjtGeom.mjGEOM_SPHERE, [s.target_radius, 0, 0], s.target_pos, [0.1, 1.0, 0.3, 0.25])]
    for p, n, f in zip(s.contact_pos, s.contact_normal, s.contact_force):
        out.append(_geom(mujoco.mjtGeom.mjGEOM_SPHERE, [0.003, 0, 0], p, [1, 0.9, 0.1, 1]))
        out.append(_arrow(p, p + n * min(0.01 + 0.01 * abs(f), 0.08), 0.002, [1, 0.9, 0.1, 1]))
    for p, n, f in zip(s.tip_pos, s.tip_normal, s.tip_force):
        if f > 0:
            out.append(_geom(mujoco.mjtGeom.mjGEOM_SPHERE, [0.006, 0, 0], p, [1, 0.15, 0.1, 1]))
            out.append(_arrow(p, p + n * min(0.015 + 0.015 * f, 0.1), 0.003, [1, 0.15, 0.1, 1]))
    return out


class Panel(plugin.ViewerPlugin):
    """Viewer side: the "Live" window (world, driver, reset, read-outs) and the 3-D overlays.

    The panel only sends requests; what it shows (selected world and driver included) is the sim's latest snapshot.
    """

    def __init__(self, n_worlds: int, drivers: Sequence[str]):
        self.n_worlds, self.drivers, self.snap = n_worlds, tuple(drivers), None

    @messages.handler
    def on_world(self, s: WorldSnapshot) -> None:
        self.snap = s

    @messages.handler
    def on_update(self, _: messages.UpdateEvent) -> None:
        self.viewer.extra_geoms[:] = overlay_geoms(self.snap) if self.snap is not None else []

    @messages.handler
    def on_build_gui(self, _: messages.BuildGuiEvent) -> None:
        s = self.snap
        imgui.SetNextWindowPos(imgui.Vec2(10, 40), imgui.Cond.FirstUseEver)
        expanded, _ = imgui.Begin("Live")
        if expanded and s is not None:
            send = self.viewer.send_to_sim
            changed, w = imgui.Combo("world", s.world, [f"world {i}" for i in range(self.n_worlds)])
            if changed:
                send(SelectWorld(w))
            imgui.SameLine()
            if imgui.Button("reset"):
                send(ResetWorld())
            for name in self.drivers:
                if imgui.RadioButton(name, s.driver == name) and s.driver != name:
                    send(SelectDriver(name))
                imgui.SameLine()
            imgui.TextUnformatted("(driver)")
            imgui.Separator()
            for line in s.lines:
                imgui.TextUnformatted(line)
            imgui.ProgressBar(s.conf, imgui.Vec2(-1, 0), f"conf_gt {s.conf:.2f}")
            for name, f in zip(FINGERS, s.tip_force):
                imgui.ProgressBar(min(f / 5.0, 1.0), imgui.Vec2(-1, 0), f"{name} {f:.2f} N")
        imgui.End()
