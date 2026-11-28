"""Scripted touch explorer: a batched state machine that makes the hand touch, grasp and sometimes pull objects.

A data-collection actor with the policy's action interface (`act(obs) -> (N, 28)` in [-1, 1]); it reads simulator
state (object poses). Arm motion uses two inverse-kinematics lattices over the zone (fingers straight down for
touching, tilted palm-down for pulls), solved once on the CPU at construction, interpolated trilinearly on the device.
Touch cycle: MOVE over an object -> DESCEND until a pad touches -> CLOSE by a random fraction -> HOLD and wiggle ->
OPEN -> RISE a little and re-grasp the same object (sometimes) or rise and MOVE to the next.
Pull cycle (probability `p_pull`): MOVE with the hand tilted -> DESCEND to the planned pinch height -> CLOSE -> HOLD
-> PULL (lift and hold: success or wrong pick ends the episode) -> OPEN -> RISE.
"""

import mujoco
import numpy as np
import torch

from tacfind.envs.bin_env import N_ARM, BinEnv
from tacfind.envs.scene import HAND

MOVE, DESCEND, CLOSE, HOLD, PULL, OPEN, RISE = range(7)
PHASES = ("move", "descend", "close", "hold", "pull", "open", "rise")


def palm_frame(rot_tcp: np.ndarray) -> np.ndarray:
    """Palm frame (3, 3) of a fingers-down tcp rotation (3, 3), base frame: columns = palm normal (the tcp x axis,
    horizontal), lateral (up x normal), up. Offsets in this frame turn with the hand (`arm_home` sets its yaw)."""
    normal = rot_tcp[:, 0] * [1.0, 1.0, 0.0]
    normal /= np.linalg.norm(normal)
    up = np.array([0.0, 0.0, 1.0])
    return np.column_stack([normal, np.cross(up, normal), up])


def tilt_rotation(tilt: float, rot_tcp: np.ndarray) -> np.ndarray:
    """Rotation (3, 3), base frame, by `tilt` rad about the horizontal axis normal x up of a fingers-down tcp rotation
    `rot_tcp` (3, 3) (negative: the palm turns to the floor)."""
    quat, rot = np.zeros(4), np.zeros(9)
    mujoco.mju_axisAngle2Quat(quat, -palm_frame(rot_tcp)[:, 1], tilt)
    mujoco.mju_quat2Mat(rot, quat)
    return rot.reshape(3, 3)


def hand_points(env: BinEnv, posture) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pinch point (3,), hand collision-mesh vertices (V, 3), both in the tcp frame, and the home tcp rotation (3, 3).

    The pinch point is the mean of the thumb pad and the index / middle pads' midpoint for a hand posture (22,).
    """
    m, d = env.mjm, mujoco.MjData(env.mjm)
    d.qpos[:] = env.qpos0.cpu().numpy()
    d.qpos[env.qadr[N_ARM:].cpu().numpy()] = posture
    mujoco.mj_kinematics(m, d)
    tcp, rot = d.site_xpos[env.tcp_site], d.site_xmat[env.tcp_site].reshape(3, 3).copy()
    pads = [m.geom(f"{HAND}right_{f}_elastomer").id for f in ("thumb", "index", "middle")]
    pinch = 0.5 * d.geom_xpos[pads[0]] + 0.25 * (d.geom_xpos[pads[1]] + d.geom_xpos[pads[2]])
    hand = np.flatnonzero((m.geom_type == mujoco.mjtGeom.mjGEOM_MESH) & (m.geom_contype == 1))  # hand collision meshes
    verts = [m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:][:m.mesh_vertnum[m.geom_dataid[g]]]
             @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g in hand]
    return (pinch - tcp) @ rot, (np.vstack(verts) - tcp) @ rot, rot


def solve_lattice(env: BinEnv, tilt: float = 0.0) -> tuple[list, np.ndarray]:
    """Arm joint angles reaching each tcp position of a lattice over the zone, tcp orientation = home tilted by `tilt`.

    Returns:
        axes: 3 coordinate arrays (x, y, z) of the lattice, m.
        q: (nx, ny, nz, 6) arm joint angles, rad (damped least squares, warm-started from the previous point).
    """
    cfg, m = env.cfg.explorer, env.mjm
    d = mujoco.MjData(m)
    d.qpos[:] = env.qpos0.cpu().numpy()
    mujoco.mj_kinematics(m, d)
    rot_home = d.site_xmat[env.tcp_site].reshape(3, 3)
    rot_goal = tilt_rotation(tilt, rot_home) @ rot_home
    half = np.asarray(env.cfg.zone.size) / 2 + cfg.lattice_margin
    axes = [np.linspace(c - h, c + h, n) for c, h, n in zip(env.cfg.zone.center, half, cfg.lattice[:2])]
    axes.append(np.linspace(*cfg.z_range, cfg.lattice[2]))
    q_arm, dofs = env.qadr[:N_ARM].cpu().numpy(), env.vadr[:N_ARM].cpu().numpy()
    lo, hi = env.q_lo[:N_ARM].cpu().numpy(), env.q_hi[:N_ARM].cpu().numpy()
    out = np.zeros((*cfg.lattice, 6))
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for ix, x in enumerate(axes[0]):
        for iy, y in enumerate(axes[1]):
            d.qpos[q_arm] = env.q_home[:N_ARM].cpu().numpy()
            for iz, z in enumerate(axes[2]):
                for _ in range(200):
                    mujoco.mj_kinematics(m, d)
                    mujoco.mj_comPos(m, d)
                    rot = d.site_xmat[env.tcp_site].reshape(3, 3)
                    err = np.concatenate([np.array([x, y, z]) - d.site_xpos[env.tcp_site],
                                          0.5 * np.cross(rot, rot_goal, axis=0).sum(1)])  # small-angle rotation error
                    mujoco.mj_jacSite(m, d, jacp, jacr, env.tcp_site)
                    jac = np.vstack([jacp[:, dofs], jacr[:, dofs]])
                    dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), np.clip(err, -0.05, 0.05))
                    d.qpos[q_arm] = np.clip(d.qpos[q_arm] + dq, lo, hi)
                    if np.linalg.norm(err) < 1e-4:
                        break
                out[ix, iy, iz] = d.qpos[q_arm]
    return axes, out


class Explorer:
    """Batched scripted explorer.

    Args:
        env: the environment (its config node `explorer` holds all parameters).
        seed: seeds goals, grasp postures, pull and re-grasp decisions.
    """

    def __init__(self, env: BinEnv, seed: int = 0):
        self.env, cfg, dev = env, env.cfg.explorer, env.device
        self.cfg, self.n = cfg, env.n
        lattices = [solve_lattice(env, 0.0), solve_lattice(env, cfg.grasp_tilt)]
        self.axes = [torch.as_tensor(a, dtype=torch.float32, device=dev) for a in lattices[0][0]]
        self.q_lattice = torch.as_tensor(np.stack([q for _, q in lattices]), dtype=torch.float32, device=dev)
        self.ik_error = self._ik_check(lattices[0][0])
        close = np.asarray(cfg.close)
        pinch, _, rot_home = hand_points(env, close)
        _, verts, _ = hand_points(env, cfg.grasp_pre * close)
        rot, palm = tilt_rotation(cfg.grasp_tilt, rot_home) @ rot_home, palm_frame(rot_home)
        self.pinch_base = torch.tensor(rot @ pinch, dtype=torch.float32, device=dev)  # tcp -> pinch, tilted, base
        as_base = lambda v: torch.tensor(palm @ np.asarray(v, float), dtype=torch.float32, device=dev)
        self.aim_offset, self.grasp_shift = as_base([*cfg.aim_offset, 0.0]), as_base(cfg.grasp_shift)  # (3,), base
        self.grasp_z_min = env.table_z + 0.008 - float((verts @ rot.T)[:, 2].min())  # hand clears the table
        self.gen = torch.Generator(dev).manual_seed(seed)
        self.dt = env.cfg.timestep * env.cfg.substeps
        self.limit = torch.tensor([round(getattr(cfg.durations, p) / self.dt) for p in PHASES], device=dev)
        self.open, self.close = torch.tensor(cfg.open, device=dev), torch.tensor(close, device=dev)
        z = lambda *s, dt=torch.float32: torch.zeros(self.n, *s, dtype=dt, device=dev)
        self.phase, self.timer, self.aim = z(dt=torch.long), z(dt=torch.long), z(dt=torch.long)
        self.pull, self.regrasp = z(dt=torch.bool), z(dt=torch.bool)
        self.goal, self.grip, self.yaw, self.q_des = z(3), z(22), z(), z(N_ARM)

    def _ik_check(self, axes) -> float:
        """Largest tcp position error of the straight-down lattice solutions, m (forward kinematics on the CPU)."""
        m, env = self.env.mjm, self.env
        d = mujoco.MjData(m)
        d.qpos[:] = env.qpos0.cpu().numpy()
        err, q_arm = 0.0, env.qadr[:N_ARM].cpu().numpy()
        for idx in np.ndindex(*self.q_lattice.shape[1:4]):
            d.qpos[q_arm] = self.q_lattice[(0, *idx)].cpu().numpy()
            mujoco.mj_kinematics(m, d)
            err = max(err, np.linalg.norm(d.site_xpos[env.tcp_site] - [axes[k][idx[k]] for k in range(3)]))
        return float(err)

    def arm_q(self, p: torch.Tensor, tilted: torch.Tensor | None = None) -> torch.Tensor:
        """Trilinear lattice interpolation: tcp positions p (N, 3) m, tilted (N,) bool -> (N, 6) arm angles, rad."""
        corner, frac = [], []
        for k, ax in enumerate(self.axes):
            u = ((p[:, k] - ax[0]) / (ax[1] - ax[0])).clamp(0, len(ax) - 1 - 1e-4)
            corner.append(u.floor().long())
            frac.append(u - u.floor())
        which = torch.zeros(len(p), dtype=torch.long, device=p.device) if tilted is None else tilted.long()
        out = 0.0
        for c in np.ndindex(2, 2, 2):
            w = torch.ones_like(p[:, 0])
            for k in range(3):
                w = w * (frac[k] if c[k] else 1 - frac[k])
            out = out + w[:, None] * self.q_lattice[which, corner[0] + c[0], corner[1] + c[1], corner[2] + c[2]]
        return out

    def _rand(self, *shape) -> torch.Tensor:
        return torch.rand(self.n, *shape, generator=self.gen, device=self.env.device)

    def _new_goal(self, mask: torch.Tensor, keep_aim: torch.Tensor | None = None) -> None:
        """Masked worlds: a spot over an object (or a random spot), wrist yaw, grasp posture, pull decision.
        Worlds in `keep_aim` re-grasp the same object from a slightly different place, at their current height."""
        cfg, env, dev = self.cfg, self.env, self.env.device
        keep = torch.zeros_like(mask) if keep_aim is None else keep_aim
        rows, center = torch.arange(self.n, device=dev), torch.tensor(env.cfg.zone.center, device=dev)
        half = torch.tensor(env.cfg.zone.size, device=dev) / 2
        pull = (self._rand() < cfg.p_pull) & ~keep
        score = self._rand(env.k) + 2.0 * env.slot_valid  # aim at an object on the table
        aim = env.active.gather(1, score.argmax(1, keepdim=True))[:, 0]
        aim = torch.where(pull & (self._rand() < cfg.pull_target), env.active_target(), aim)  # some pulls: the target
        self.aim = torch.where(mask & ~keep, aim, self.aim)
        aim_xy = env.sim.xpos[rows, env.obj_body[self.aim], :2]
        noise = (self._rand(2) * 2 - 1) * cfg.aim_noise
        xy = torch.where(self._rand(1) < cfg.p_aim, aim_xy + self.aim_offset[:2] + noise,
                         center + (self._rand(2) * 2 - 1) * half)
        reach = half + cfg.lattice_margin
        xy = torch.where(pull[:, None], aim_xy + self.grasp_shift[:2] - self.pinch_base[:2],
                         xy).clamp(center - reach, center + reach)
        z = torch.where(keep, env.sim.site_xpos[:, env.tcp_site, 2], torch.full((self.n,), cfg.hover_z, device=dev))
        self.goal = torch.where(mask[:, None], torch.cat([xy, z[:, None]], 1), self.goal)
        self.pull = torch.where(mask, pull, self.pull)
        self.yaw = torch.where(mask, (self._rand() * 2 - 1) * cfg.max_yaw * ~pull, self.yaw)
        lo, hi = cfg.close_frac
        frac = (lo + (hi - lo) * self._rand(5)).repeat_interleave(torch.tensor([5, 4, 4, 4, 5], device=dev), 1)
        grip = torch.where(pull[:, None], self.close, self.open + frac * (self.close - self.open))
        self.grip = torch.where(mask[:, None], grip, self.grip)

    def act(self, obs: dict) -> torch.Tensor:
        """Actions (N, 28) in [-1, 1] for the current observation (uses obs["first"], obs["pts_valid"])."""
        cfg, env, dev = self.cfg, self.env, self.env.device
        first = obs["first"].to(dev)
        self.phase, self.timer = torch.where(first, MOVE, self.phase), torch.where(first, 0, self.timer)
        self._new_goal(first)
        tcp = env.sim.site_xpos[:, env.tcp_site]
        contact = obs["pts_valid"].to(dev).any(1)
        ph, t, pull = self.phase, self.timer, self.pull
        done = t >= self.limit[ph]
        # reached: tcp at its goal and the arm at its joint targets (covers the wrist turning to / from the tilt)
        reached = ((tcp - self.goal).norm(dim=1) < 0.01) & ((env.sim.qpos[:, env.qadr[:N_ARM]] - self.q_des).abs()
                                                            .amax(1) < 0.05)
        at_floor = self.goal[:, 2] <= torch.where(pull, self.grasp_low_z(), torch.full_like(t, cfg.descend_min_z,
                                                                                             dtype=torch.float)) + 1e-4
        nxt = torch.where((ph == MOVE) & (reached | done), DESCEND, ph)
        nxt = torch.where((ph == DESCEND) & ((contact & ~pull) | (at_floor & (reached | ~pull)) | done), CLOSE, nxt)
        nxt = torch.where((ph == CLOSE) & done, HOLD, nxt)
        nxt = torch.where((ph == HOLD) & done, torch.where(pull, PULL, OPEN), nxt)
        nxt = torch.where((ph == PULL) & done, OPEN, nxt)
        nxt = torch.where((ph == OPEN) & done, RISE, nxt)
        nxt = torch.where((ph == RISE) & (reached | done), torch.where(self.regrasp, DESCEND, MOVE), nxt)
        changed = nxt != ph
        self.regrasp = torch.where(changed & (nxt == RISE), (self._rand() < cfg.p_regrasp) & ~pull, self.regrasp)
        self._new_goal(changed & (nxt == MOVE))
        self._new_goal(changed & (ph == RISE) & (nxt == DESCEND), keep_aim=torch.ones_like(changed))
        self.phase, self.timer = nxt, torch.where(changed, 0, t + 1)
        self._update_goal_height(changed, tcp)
        ph = self.phase
        hold = (ph == HOLD) & ~self.pull
        wiggle = cfg.wiggle * torch.sin(self.timer[:, None] * 0.8 + torch.arange(3, device=dev)) * hold[:, None]
        q_arm = self.arm_q(self.goal + wiggle, self.pull)
        q_arm[:, 5] += self.yaw
        self.q_des = q_arm
        closed = (ph == CLOSE) | (ph == HOLD) | (ph == PULL)
        pre = torch.where(self.pull[:, None], cfg.grasp_pre * self.close, self.open)
        q_hand = torch.where(closed[:, None], self.grip, pre)
        return ((torch.cat([q_arm, q_hand], 1) - env.q_target) / env.step_size).clamp(-1, 1)

    def grasp_low_z(self) -> torch.Tensor:
        """(N,) tcp height that puts the pinch point at the aimed object's centre + shift, hand clear of the floor."""
        env = self.env
        z_obj = env.sim.xpos[torch.arange(self.n, device=env.device), env.obj_body[self.aim], 2]
        return (z_obj + self.grasp_shift[2] - self.pinch_base[2]).clamp(min=self.grasp_z_min)

    def _update_goal_height(self, changed: torch.Tensor, tcp: torch.Tensor) -> None:
        """Per-phase tcp goal height: descend gradually (to contact, or to the planned grasp height), pull, rise."""
        cfg, ph, gz = self.cfg, self.phase, self.goal[:, 2]
        floor = torch.where(self.pull, self.grasp_low_z(), torch.full_like(gz, cfg.descend_min_z))
        gz = torch.where(changed & (ph == DESCEND), tcp[:, 2], gz)  # descend from where the hand actually is
        gz = torch.where(ph == DESCEND, torch.maximum(gz - cfg.descend_speed * self.dt, floor), gz)
        gz = torch.where(ph == PULL, torch.full_like(gz, cfg.pull_z), gz)
        rise_to = torch.where(self.regrasp, (tcp[:, 2] + cfg.regrasp_rise).clamp(max=cfg.hover_z),
                              torch.full_like(gz, cfg.hover_z))
        gz = torch.where(changed & (ph == RISE), rise_to, gz)
        self.goal = torch.cat([self.goal[:, :2], gz[:, None]], 1)
