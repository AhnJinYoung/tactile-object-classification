"""Grasp feasibility check (privileged, CPU MuJoCo): can the simulated hand press hard enough and lift each object?

Not part of the training pipeline: a diagnostic for the scene and the finger gains.
    python -m tacfind.envs.grasp_check                     # press forces + lift table for both finger_gains settings
Each trial: one library object alone at the zone centre; the tilted hand moves over it, descends, closes to the
pinch posture (`explorer.close`), lifts and holds; success = the object's lowest point stays `lift_height` above the
table for the whole `hold_time`, as the env's success rule.
"""

import copy

import mujoco
import numpy as np
from omegaconf import OmegaConf

from tacfind.envs.bin_env import REPO, plain
from tacfind.envs.explorer import palm_frame, tilt_rotation
from tacfind.envs.objects import build_library
from tacfind.envs.scene import ARM_JOINTS, HAND, build_scene, hand_spec


def press_force(cfg, finger: str = "index", flex: float = 1.0) -> float:
    """Steady contact force (N) of a fingertip pressing a rigid block, the finger commanded `flex` rad past touch."""
    spec = mujoco.MjSpec()
    spec.option.timestep, spec.option.cone, spec.option.impratio = cfg.timestep, mujoco.mjtCone.mjCONE_ELLIPTIC, 10
    frame = spec.worldbody.add_frame()
    spec.attach(hand_spec(cfg, REPO / cfg.assets), frame=frame, prefix=HAND)
    for body in spec.bodies:
        body.gravcomp = 1.0
    # block in front of the fingertip pads (hand frame: fingers along +z, pads facing +x)
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.08, 0.03], pos=[0.03, 0.03, 0.21])
    m = spec.compile()
    d = mujoco.MjData(m)
    names = [m.actuator(a).name for a in range(m.nu)]
    for a, name in enumerate(names):  # flex the chosen finger's FE joints; the others stay straight
        if f"_{finger}_" in name and ("FE" in name or "PIP" in name or "DIP" in name or name.endswith("_IP_ctrl")):
            d.ctrl[a] = flex
    mujoco.mj_step(m, d, nstep=round(1.5 / cfg.timestep))
    tip = m.body(f"{HAND}right_{finger}_DP").id
    f6, total = np.zeros(6), np.zeros(3)
    for c in range(d.ncon):
        if tip in m.geom_bodyid[d.contact.geom[c]]:
            mujoco.mj_contactForce(m, d, c, f6)
            total += d.contact.frame[c].reshape(3, 3).T @ f6[:3]
    return float(np.linalg.norm(total))


class LiftTrial:
    """Scripted grasp-and-lift of one library object (arm by damped least-squares IK on the tcp site).

    The grasp is planned in the tcp frame: `close` defines the pinch point (mean of the thumb, index and middle pad
    centres); the tcp is placed so that this point lands at the object's centre + `shift` (palm frame, see
    `explorer.palm_frame`), but no lower than lets the hand (in posture `pre`) clear the table by `clearance`. `tilt`
    rotates the hand about the horizontal axis across the palm (negative: the palm turns towards the table), applied
    for the whole trial.
    """

    def __init__(self, cfg, obj):
        self.cfg, self.obj = cfg, obj
        self.m = m = build_scene(cfg, REPO / cfg.assets, [obj])
        self.d = mujoco.MjData(m)
        self.arm_q = np.array([m.jnt_qposadr[m.joint(f"{j}_joint").id] for j in ARM_JOINTS])
        self.arm_v = np.array([m.jnt_dofadr[m.joint(f"{j}_joint").id] for j in ARM_JOINTS])
        self.hand_q = m.jnt_qposadr[m.actuator_trnid[6:, 0]]
        self.tcp = m.site(f"{HAND}tcp").id
        self.obj_body, self.obj_q = m.body("obj0").id, m.jnt_qposadr[m.joint("obj0").id]
        self.dt = cfg.timestep * cfg.substeps
        home = np.asarray(cfg.arm_home)
        self.q_lo, self.q_hi = home - np.asarray(cfg.arm_range), home + np.asarray(cfg.arm_range)

    def lowest(self) -> float:
        """Lowest point of the object (cloud), base z, m."""
        rot = self.d.xmat[self.obj_body].reshape(3, 3)
        return float((self.obj.cloud[:, :3] @ rot.T)[:, 2].min() + self.d.xpos[self.obj_body, 2])

    def hand_points(self, posture) -> tuple[np.ndarray, np.ndarray]:
        """Pinch point (3,) and all hand collision-mesh vertices (V, 3) for a posture, in the tcp frame, m."""
        m, d = self.m, mujoco.MjData(self.m)
        d.qpos[self.arm_q], d.qpos[self.hand_q] = self.cfg.arm_home, posture
        mujoco.mj_kinematics(m, d)
        to_tcp = lambda x: (x - d.site_xpos[self.tcp]) @ d.site_xmat[self.tcp].reshape(3, 3)
        pads = [m.geom(f"{HAND}right_{f}_elastomer").id for f in ("thumb", "index", "middle")]
        pinch = to_tcp(0.5 * d.geom_xpos[pads[0]] + 0.25 * (d.geom_xpos[pads[1]] + d.geom_xpos[pads[2]]))
        verts = []
        for g in np.flatnonzero((m.geom_type == mujoco.mjtGeom.mjGEOM_MESH) & (m.geom_contype == 1)):  # hand meshes
            v = m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:][:m.mesh_vertnum[m.geom_dataid[g]]]
            verts.append(to_tcp(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]))
        return pinch, np.vstack(verts)

    def _ik(self, goal: np.ndarray, rot_goal: np.ndarray) -> np.ndarray:
        """Arm joint targets one DLS step towards tcp pose (goal, rot_goal), clipped to the env's arm range."""
        m, d = self.m, self.d
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, self.tcp)
        jac = np.vstack([jacp[:, self.arm_v], jacr[:, self.arm_v]])
        rot = d.site_xmat[self.tcp].reshape(3, 3)
        err = np.concatenate([goal - d.site_xpos[self.tcp], 0.5 * np.cross(rot, rot_goal, axis=0).sum(1)])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), np.clip(err, -0.05, 0.05))
        return np.clip(d.qpos[self.arm_q] + dq, self.q_lo, self.q_hi)

    def run(self, rng: np.random.Generator, pre, close, tilt: float, shift=(0.0, 0.0, 0.0),
            clearance: float = 0.008) -> dict:
        """One trial with the object alone at the zone centre (random orientation in half the trials).

        Returns:
            {"success": lowest point >= lift_height above the table during the whole final hold of `hold_time` s,
             "max_height": highest lift of the lowest point above the table (m; ~0: never lifted, else slipped),
             "reach_err": largest tcp error at the end of a phase (m)}.
        """
        cfg, m, d = self.cfg, self.m, self.d
        ex, top = cfg.explorer, cfg.table.top_z
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_q] = d.ctrl[:6] = cfg.arm_home
        d.qpos[self.hand_q] = d.ctrl[6:] = pre
        quat = rng.normal(size=4) if rng.uniform() < 0.5 else np.array([1.0, 0, 0, 0])
        d.qpos[self.obj_q:self.obj_q + 7] = [*cfg.zone.center, top + 0.08, *(quat / np.linalg.norm(quat))]
        mujoco.mj_step(m, d, nstep=round(0.8 / cfg.timestep))
        rot_home = d.site_xmat[self.tcp].reshape(3, 3).copy()
        rot_goal = tilt_rotation(tilt, rot_home) @ rot_home
        pinch, _ = self.hand_points(close)
        _, verts = self.hand_points(pre)
        low = d.xpos[self.obj_body] + palm_frame(rot_home) @ np.asarray(shift) - rot_goal @ pinch  # pinch target
        low[2] = max(low[2], top + clearance - (verts @ rot_goal.T)[:, 2].min())
        above, high = low + [0, 0, ex.hover_z - low[2]], low + [0, 0, ex.pull_z - low[2]]
        plan = [(1.5, above, pre), (1.5, low, pre), (1.0, low, close), (2.0, high, close), (cfg.hold_time, high, close)]
        start, hand0, reach, best, held = d.site_xpos[self.tcp].copy(), np.asarray(pre, float), 0.0, 0.0, True
        for phase, (duration, goal, hand) in enumerate(plan):
            n = round(duration / self.dt)
            for k in range(1, n + 1):  # linear interpolation of tcp goal and hand posture over the phase
                u = k / n
                d.ctrl[:6] = self._ik(start + u * (goal - start), rot_goal)
                d.ctrl[6:] = hand0 + u * (np.asarray(hand) - hand0)
                mujoco.mj_step(m, d, nstep=cfg.substeps)
                best = max(best, self.lowest() - top)
                held &= phase < 4 or self.lowest() - top >= cfg.lift_height
            reach = max(reach, float(np.linalg.norm(d.site_xpos[self.tcp] - goal)))
            start, hand0 = goal, np.asarray(hand, float)
        return {"success": held, "max_height": best, "reach_err": reach}


def lift_table(cfg, trials: int = 3, seed: int = 0) -> dict:
    """Lift success per library object: {name: (successes, trials, mass g, longest extent cm, failures)}, where
    failures counts trials that never lifted the object (0.01 m) and trials where it slipped out after lifting."""
    rng, out, ex = np.random.default_rng(seed), {}, cfg.explorer
    close = np.asarray(ex.close)
    for obj in build_library(cfg.objects):
        trial = LiftTrial(cfg, obj)
        res = [trial.run(rng, ex.grasp_pre * close, close, ex.grasp_tilt, ex.grasp_shift) for _ in range(trials)]
        fails = [r for r in res if not r["success"]]
        out[obj.family] = (len(res) - len(fails), trials, 1000 * obj.mass, 100 * np.ptp(obj.cloud[:, :3], 0).max(),
                           {"not lifted": sum(r["max_height"] < 0.01 for r in fails),
                            "slipped": sum(r["max_height"] >= 0.01 for r in fails)})
    return out


if __name__ == "__main__":
    base = plain(OmegaConf.load(REPO / "configs" / "env" / "bin.yaml"))
    base.randomize.enabled = False
    for gains in ("mjcf", "torque_limit"):
        cfg = copy.deepcopy(base)
        cfg.finger_gains = gains
        forces = {flex: press_force(cfg, "index", flex) for flex in (0.2, 0.5, 1.0, 1.5)}
        print(f"[{gains}] index fingertip on a rigid block vs commanded flexion: "
              + ", ".join(f"{k} rad {v:.1f} N" for k, v in forces.items()) + f"; max {max(forces.values()):.1f} N")
        table = lift_table(cfg)
        for name, (ok, n, mass, ext, fails) in table.items():
            print(f"  {name:15s} {mass:5.0f} g {ext:5.1f} cm  lifted {ok}/{n}  {fails}", flush=True)
        held = [(mass, name) for name, (ok, n, mass, *_) in table.items() if ok >= 2 * n / 3]
        total = sum(v[0] for v in table.values()), sum(v[1] for v in table.values())
        print(f"  [{gains}] lifted {total[0]}/{total[1]}; heaviest object held in >= 2/3 trials: "
              f"{max(held)[1]} ({max(held)[0]:.0f} g)" if held else f"  [{gains}] nothing held", flush=True)
