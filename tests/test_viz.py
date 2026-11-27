"""Live viewer command handling (tacfind/viz/live.py) on the CPU backend, without opening a socket."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from mujoco.studio import messages, plugin_registry, sim

from tacfind.envs.bin_env import BinEnv
from tacfind.viz.live import LiveViewer
from tacfind.viz.panel import ResetWorld, SelectDriver, SelectWorld, WorldSnapshot, overlay_geoms


@pytest.fixture
def setup(cfg):
    env = BinEnv(cfg, 2, history=30, device="cpu", seed=0)
    env.reset()
    viewer = LiveViewer(env, drivers=("explorer", "policy"))
    sent = []
    viewer.handle = SimpleNamespace(send_to_viewer=sent.append)  # stands in for the web server
    calls = []
    env.perturb = lambda w, f: calls.append((w, f))  # records the calls instead of pushing bodies
    registry = plugin_registry.PluginRegistry([viewer.bridge])
    yield env, viewer, registry, sent, calls
    env.sim.pool.shutdown()


def browser(viewer: LiveViewer, registry, *msgs):
    """Deliver messages as `ViewerHandle.sync` does, then run the bookkeeping of `LiveViewer.sync`."""
    for m in msgs:
        registry.dispatch(m)
    cmd = viewer.bridge.take()
    viewer._perturb(cmd.world)
    viewer._select(cmd)
    return cmd


def ctrl_event(model: mujoco.MjModel, ctrl: np.ndarray) -> messages.StateEvent:
    d = mujoco.MjData(model)
    d.ctrl[:] = ctrl
    return messages.StateEvent.from_data(model, d, int(mujoco.mjtState.mjSTATE_CTRL))


def test_step_control_resets_and_world(setup):
    env, viewer, reg, _, _ = setup
    cmd = browser(viewer, reg, messages.StepControlSnapshot(pause_state=sim.PauseState.NORMAL_PAUSED, speed=25.0),
                  messages.SingleStepEvent(), messages.SingleStepEvent(), SelectWorld(1), ResetWorld())
    assert cmd.paused and cmd.speed == 25.0 and cmd.steps == 2 and cmd.world == 1 and cmd.resets == [1]
    cmd = browser(viewer, reg, messages.ResetEvent(key=None), SelectWorld(7))  # out of range: ignored
    assert cmd.steps == 0 and cmd.resets == [1] and cmd.world == 1 and viewer.world == 1


def test_drivers_map_to_override(setup):
    env, viewer, reg, _, _ = setup
    browser(viewer, reg, SelectWorld(1), SelectDriver("hold"))
    viewer.drive()
    assert env.override_mask.tolist() == [False, True] and env.override_action[1].abs().max() == 0
    q_des = env.q_target[1].numpy().copy()
    q_des[0] += 0.01  # a third of the arm step
    q_des[10] -= 1.0  # far: saturates
    browser(viewer, reg, ctrl_event(env.mjm, q_des))
    assert viewer.driver == "manual"  # a slider edit takes the world over
    viewer.drive()
    a = env.override_action[1]
    assert a[0] == pytest.approx(0.01 / env.cfg.arm_step, rel=1e-4) and a[10] == -1.0
    q0 = env.q_target[1, 0].item()
    for _ in range(3):
        viewer.drive()
        env.step(torch.zeros(2, 28))
    assert env.q_target[1, 0].item() == pytest.approx(q0 + 0.01, abs=1e-5)  # reached and held
    policy = torch.full((2, 28), 0.5)
    browser(viewer, reg, SelectDriver("policy"))
    viewer.drive({"explorer": torch.zeros(2, 28), "policy": policy})
    assert env.override_mask[1] and torch.equal(env.override_action[1], policy[1])
    browser(viewer, reg, SelectDriver("explorer"))
    viewer.drive({"explorer": torch.zeros(2, 28)})
    assert not env.override_mask.any()
    browser(viewer, reg, SelectDriver("hold"), SelectWorld(0))  # switching hands the old world back
    viewer.drive()
    assert env.override_mask.tolist() == [True, False]


def test_perturbation_and_refused_events(setup):
    env, viewer, reg, _, calls = setup
    sig = int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)
    x = np.zeros(mujoco.mj_stateSize(env.mjm, sig))
    x[6 * 5 + 2] = 4.0
    for _ in range(3):
        browser(viewer, reg, messages.PerturbSnapshot(state=x, state_sig=sig))
    assert len(calls) == 1 and calls[0][0] == 0 and calls[0][1].shape == (env.mjm.nbody, 6)
    assert calls[0][1][5, 2] == 4.0
    browser(viewer, reg, messages.PerturbSnapshot(state=x, state_sig=sig), SelectWorld(1))
    assert calls[-2] == (0, None) and calls[-1][0] == 1  # cleared on the old world, then sent to the new one
    browser(viewer, reg, messages.PerturbSnapshot(state=np.zeros_like(x), state_sig=sig))
    assert calls[-1] == (1, None)

    later = []

    class Loader:  # stands in for ViewerHandle, whose INTERNAL handler would load the file
        @messages.handler(priority=messages.Priority.INTERNAL)
        def load(self, e: messages.LoadModelEvent) -> None:
            later.append(e)

    reg2 = plugin_registry.PluginRegistry([viewer.bridge, Loader()])
    reg2.dispatch(messages.LoadModelEvent(path="dropped.xml"))
    reg2.dispatch(messages.LoadModelEvent(path=""))
    assert later == []


def test_view_state_and_overlays(setup):
    env, viewer, reg, sent, _ = setup
    env.step(torch.zeros(2, 28))
    browser(viewer, reg, SelectWorld(1))
    viewer._show()
    s = sent[-1]
    assert isinstance(s, WorldSnapshot) and s.world == 1 and s.driver == "explorer"
    np.testing.assert_allclose(viewer.data.qpos, env.world_state(1)["qpos"], atol=1e-6)
    np.testing.assert_allclose(viewer.data.ctrl, env.q_target[1].numpy(), atol=1e-6)
    tgt = env.obj_body[env.active_target()[1]].item()
    np.testing.assert_allclose(s.target_pos, env.sim.xpos[1, tgt].numpy(), atol=1e-4)
    viewer._show()
    assert len(sent) == 1  # nothing new: nothing sent
    geoms = overlay_geoms(s)
    assert len(geoms) == 1 + 2 * len(s.contact_pos) + 2 * int((s.tip_force > 0).sum())
