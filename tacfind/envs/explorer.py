"""Scripted touch explorer: a batched state machine that makes the hand touch, grasp and sometimes pull objects.

A data-collection actor with the policy's action interface (`act(obs) -> (N, 28)` in [-1, 1]); it may read
simulator state. Arm motion uses an inverse-kinematics lattice over the bin, solved once on the CPU at construction
and interpolated trilinearly on the simulation device.
Cycle per world: MOVE over a random spot -> DESCEND until a pad touches (or the floor is near) -> CLOSE the fingers
by a random fraction -> HOLD and wiggle -> PULL (sometimes) -> OPEN -> RISE -> MOVE ...
"""

import mujoco
import numpy as np
import torch

from tacfind.envs.bin_env import N_ARM, BinEnv

MOVE, DESCEND, CLOSE, HOLD, PULL, OPEN, RISE = range(7)
PHASES = ("move", "descend", "close", "hold", "pull", "open", "rise")


def solve_lattice(env: BinEnv) -> tuple[np.ndarray, np.ndarray]:
    """Arm joint angles reaching each tcp position of a lattice over the bin with the home tcp orientation.

    Returns:
        axes: list of 3 coordinate arrays (x, y, z) of the lattice, m.
        q: (nx, ny, nz, 6) arm joint angles, rad (damped least squares, warm-started from the previous point).
    """
    cfg, m = env.cfg.explorer, env.mjm
    d = mujoco.MjData(m)
    d.qpos[:] = env.qpos0.cpu().numpy()
    mujoco.mj_kinematics(m, d)
    rot_home = d.site_xmat[env.tcp_site].reshape(3, 3).copy()
    cx, cy = env.cfg.bin.center
    axes = [np.linspace(c - cfg.xy_range, c + cfg.xy_range, n) for c, n in zip((cx, cy), cfg.lattice[:2])]
    axes.append(np.linspace(*cfg.z_range, cfg.lattice[2]))
    q_arm = env.qadr[:N_ARM].cpu().numpy()
    dofs = env.vadr[:N_ARM].cpu().numpy()
    lo, hi = env.q_lo[:N_ARM].cpu().numpy(), env.q_hi[:N_ARM].cpu().numpy()
    out = np.zeros((*cfg.lattice, 6))
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for ix, x in enumerate(axes[0]):
        for iy, y in enumerate(axes[1]):
            d.qpos[q_arm] = env.q_home[:N_ARM].cpu().numpy()
            for iz, z in enumerate(axes[2]):
                for _ in range(100):
                    mujoco.mj_kinematics(m, d)
                    mujoco.mj_comPos(m, d)
                    err_p = np.array([x, y, z]) - d.site_xpos[env.tcp_site]
                    rot = d.site_xmat[env.tcp_site].reshape(3, 3)
                    err_r = 0.5 * np.cross(rot, rot_home, axis=0).sum(1)  # small-angle orientation error
                    mujoco.mj_jacSite(m, d, jacp, jacr, env.tcp_site)
                    jac = np.vstack([jacp[:, dofs], jacr[:, dofs]])
                    err = np.concatenate([err_p, err_r])
                    dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
                    d.qpos[q_arm] = np.clip(d.qpos[q_arm] + dq, lo, hi)
                    if np.linalg.norm(err) < 1e-4:
                        break
                out[ix, iy, iz] = d.qpos[q_arm]
    return axes, out


class Explorer:
    """Batched scripted explorer.

    Args:
        env: the environment (its config node `explorer` holds all parameters).
        seed: seeds goals, grasp postures and pull decisions.
    """

    def __init__(self, env: BinEnv, seed: int = 0):
        self.env, cfg, dev = env, env.cfg.explorer, env.device
        self.cfg, self.n = cfg, env.n
        axes, q = solve_lattice(env)
        self.axes = [torch.as_tensor(a, dtype=torch.float32, device=dev) for a in axes]
        self.q_lattice = torch.as_tensor(q, dtype=torch.float32, device=dev)
        self.ik_error = self._ik_check(axes)
        self.gen = torch.Generator(dev).manual_seed(seed)
        self.dt = env.cfg.timestep * env.cfg.substeps
        self.limit = {p: round(getattr(cfg.durations, p) / self.dt) for p in PHASES}
        self.open = torch.tensor(cfg.open, device=dev)
        self.close = torch.tensor(cfg.close, device=dev)
        z = lambda *s, dt=torch.float32: torch.zeros(self.n, *s, dtype=dt, device=dev)
        self.phase, self.timer, self.pull = z(dt=torch.long), z(dt=torch.long), z(dt=torch.bool)
        self.goal, self.grip, self.yaw = z(3), z(22), z()

    def _ik_check(self, axes) -> float:
        """Largest tcp position error of the lattice solutions, m (forward kinematics on the CPU)."""
        m, env = self.env.mjm, self.env
        d = mujoco.MjData(m)
        d.qpos[:] = env.qpos0.cpu().numpy()
        err = 0.0
        q_arm = env.qadr[:N_ARM].cpu().numpy()
        for idx in np.ndindex(*self.q_lattice.shape[:3]):
            d.qpos[q_arm] = self.q_lattice[idx].cpu().numpy()
            mujoco.mj_kinematics(m, d)
            err = max(err, np.linalg.norm(d.site_xpos[env.tcp_site] - [axes[k][idx[k]] for k in range(3)]))
        return float(err)

    def arm_q(self, p: torch.Tensor) -> torch.Tensor:
        """Trilinear interpolation of the IK lattice at tcp positions p (N, 3) m -> (N, 6) arm joint angles, rad."""
        corner, frac = [], []
        for k, ax in enumerate(self.axes):
            u = ((p[:, k] - ax[0]) / (ax[1] - ax[0])).clamp(0, len(ax) - 1 - 1e-4)
            corner.append(u.floor().long())
            frac.append(u - u.floor())
        out = 0.0
        for c in np.ndindex(2, 2, 2):
            w = torch.ones_like(p[:, 0])
            for k in range(3):
                w = w * (frac[k] if c[k] else 1 - frac[k])
            out = out + w[:, None] * self.q_lattice[corner[0] + c[0], corner[1] + c[1], corner[2] + c[2]]
        return out

    def _new_goal(self, mask: torch.Tensor) -> None:
        """Masked worlds: a random spot over the bin, a random wrist yaw, a random grasp posture and pull decision."""
        cfg, n, dev = self.cfg, self.n, self.env.device
        rand = lambda *s: torch.rand(n, *s, generator=self.gen, device=dev)
        center = torch.tensor(self.env.cfg.bin.center, device=dev)
        xy = center + (rand(2) * 2 - 1) * cfg.xy_range
        env, rows = self.env, torch.arange(n, device=dev)
        aim = env.active[rows, torch.randint(env.cfg.n_slots, (n,), generator=self.gen, device=dev)]
        aim_xy = env.sim.xpos[rows, env.obj_body[aim], :2] + torch.tensor(cfg.aim_offset, device=dev)
        aim_xy = aim_xy + (rand(2) * 2 - 1) * cfg.aim_noise
        xy = torch.where(rand(1) < cfg.p_aim, aim_xy, xy).clamp(center - cfg.xy_range, center + cfg.xy_range)
        hover = torch.full((n, 1), cfg.hover_z, device=dev)
        self.goal = torch.where(mask[:, None], torch.cat([xy, hover], 1), self.goal)
        self.yaw = torch.where(mask, (rand() * 2 - 1) * cfg.max_yaw, self.yaw)
        lo, hi = cfg.close_frac
        frac = (lo + (hi - lo) * rand(5)).repeat_interleave(torch.tensor([5, 4, 4, 4, 5], device=dev), 1)
        self.grip = torch.where(mask[:, None], self.open + frac * (self.close - self.open), self.grip)
        self.pull = torch.where(mask, rand() < cfg.p_pull, self.pull)

    def act(self, obs: dict) -> torch.Tensor:
        """Actions (N, 28) in [-1, 1] for the current observation (uses obs["first"], obs["pts_valid"])."""
        cfg, env, dev = self.cfg, self.env, self.env.device
        first = obs["first"].to(dev)
        self.phase = torch.where(first, MOVE, self.phase)
        self.timer = torch.where(first, 0, self.timer)
        self._new_goal(first)
        tcp = env.sim.site_xpos[:, env.tcp_site]
        contact = obs["pts_valid"].to(dev).any(1)
        ph, t = self.phase, self.timer
        limit = torch.tensor([self.limit[p] for p in PHASES], device=dev)[ph]
        reached = (tcp - self.goal).norm(dim=1) < 0.01
        z_floor = self.goal[:, 2] <= cfg.z_range[0] + 1e-4
        nxt = torch.where(ph == MOVE, torch.where(reached | (t >= limit), DESCEND, MOVE), ph)
        nxt = torch.where(ph == DESCEND, torch.where(contact | z_floor | (t >= limit), CLOSE, DESCEND), nxt)
        nxt = torch.where((ph == CLOSE) & (t >= limit), HOLD, nxt)
        nxt = torch.where((ph == HOLD) & (t >= limit), torch.where(self.pull, PULL, OPEN), nxt)
        nxt = torch.where((ph == PULL) & (t >= limit), OPEN, nxt)
        nxt = torch.where((ph == OPEN) & (t >= limit), RISE, nxt)
        nxt = torch.where((ph == RISE) & (reached | (t >= limit)), MOVE, nxt)
        changed = nxt != ph
        self._new_goal(changed & (nxt == MOVE))
        self.phase, self.timer = nxt, torch.where(changed, 0, t + 1)
        ph = self.phase
        # tcp goal height per phase
        gz = self.goal[:, 2]
        gz = torch.where(ph == DESCEND, (gz - cfg.descend_speed * self.dt).clamp(min=cfg.z_range[0]), gz)
        gz = torch.where(changed & (ph == DESCEND), tcp[:, 2], gz)  # descend from where the hand actually is
        gz = torch.where(ph == PULL, torch.full_like(gz, cfg.pull_z), gz)
        gz = torch.where(ph == RISE, torch.full_like(gz, cfg.hover_z), gz)
        self.goal = torch.cat([self.goal[:, :2], gz[:, None]], 1)
        wiggle = cfg.wiggle * torch.sin(self.timer[:, None] * 0.8 + torch.arange(3, device=dev)) * (ph == HOLD)[:, None]
        q_arm = self.arm_q(self.goal + wiggle)
        q_arm[:, 5] += self.yaw
        closed = (ph == CLOSE) | (ph == HOLD) | (ph == PULL)
        q_hand = torch.where(closed[:, None], self.grip, self.open.expand(self.n, -1))
        q_des = torch.cat([q_arm, q_hand], 1)
        return ((q_des - env.q_target) / env.step_size).clamp(-1, 1)
