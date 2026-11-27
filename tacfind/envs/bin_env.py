"""Batched bin environment: blind touch search for a target object, then pull it out of the bin.

Sits between the physics backend (`sim.py`, `scene.py`, `tactile.py`) and the trainer: `BinEnv.step` turns (N, 28)
joint-target increments into observations for the networks, privileged labels for the losses (`info`) and a shaped
reward. Every quantity is a torch tensor on the simulation device; there are no per-world Python loops in the step path.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from tacfind.envs.objects import build_library, split_indices
from tacfind.envs.scene import ARM_JOINTS, HAND, build_scene, park_position
from tacfind.envs.sim import make_sim
from tacfind.envs.tactile import N_TIPS, Tactile
from tacfind.spec import EnvSpec

REPO = Path(__file__).resolve().parents[2]
N_ARM, N_HAND = 6, 22
GROUP_SIZES = {"arm_q": 6, "arm_qd": 6, "hand_q": 22, "hand_qd": 22, "q_target": 28, "tcp_pose": 9, "tcp_wrench": 6,
               "tip_wrench": 30, "tip_force": 5, "tip_contact": 5, "tip_point": 15, "tip_normal": 15, "prev_action": 28}
PRIV_PER_SLOT = 10  # object position relative to the tcp (3), 6-D rotation (6), is-target flag (1)


def plain(node):
    """Config node (OmegaConf or dict) -> nested SimpleNamespace / list: OmegaConf attribute access is slow."""
    if isinstance(node, Mapping):
        return SimpleNamespace(**{k: plain(v) for k, v in node.items()})
    if isinstance(node, Sequence) and not isinstance(node, str):
        return [plain(v) for v in node]
    return node


def rot6d(mat: torch.Tensor) -> torch.Tensor:
    """First two columns of rotation matrices (..., 3, 3) -> (..., 6)."""
    return mat[..., :, :2].transpose(-1, -2).reshape(*mat.shape[:-2], 6)


class BinEnv:
    """N parallel bin worlds in one batched simulation.

    Args:
        cfg: the `env` config node (configs/env/bin.yaml).
        n_envs: number N of worlds.
        history: H, steps of contact memory behind the coverage label.
        device: simulation device ("cuda", "cuda:1", "cpu"); the cpu backend always runs on "cpu".
        seed: seeds the settled-state bank and every reset.
        split: "train" or "heldout" library objects.
    """

    def __init__(self, cfg, n_envs: int, history: int, device: str = "cuda", seed: int = 0, split: str = "train"):
        self.cfg = cfg = plain(cfg)
        self.n, self.history = n_envs, history
        lib = build_library(cfg.objects)
        self.lib_ids = torch.as_tensor(split_indices(cfg.objects, split))  # library index of object body j
        self.mjm = build_scene(cfg, (REPO / cfg.assets).resolve(), [lib[i] for i in self.lib_ids.tolist()])
        self.sim = make_sim(cfg.sim, self.mjm, n_envs, device, nconmax=cfg.nconmax, njmax=cfg.njmax,
                            n_threads=cfg.n_threads)
        dev = self.device = self.sim.device
        self.clouds = torch.as_tensor(np.stack([o.cloud for o in lib]), device=dev)  # (n_lib, N_pts, 6)
        self.lib_ids = self.lib_ids.to(dev)
        self.tactile = Tactile(self.mjm, cfg.force_threshold, dev)
        self.gen = torch.Generator(dev).manual_seed(seed)
        self._init_constants()
        self.spec = EnvSpec(obs_dim=sum(GROUP_SIZES[g] for g in cfg.obs_groups), act_dim=N_ARM + N_HAND,
                            priv_dim=PRIV_PER_SLOT * cfg.n_slots, n_tips=N_TIPS, pt_dim=7)
        z = lambda *s, dt=torch.float32: torch.zeros(n_envs, *s, dtype=dt, device=dev)
        self.q_target, self.prev_action, self.t = z(N_ARM + N_HAND), z(N_ARM + N_HAND), z(dt=torch.long)
        self.active, self.target = z(cfg.n_slots, dt=torch.long), z(dt=torch.long)  # object bodies j; target slot
        self.target_z, self.last_touch = z(), z(self.clouds.shape[1], dt=torch.long)
        self.override_mask, self.override_action = z(dt=torch.bool), z(N_ARM + N_HAND)
        self._build_bank(cfg.bank_size)
        self.last = {}

    def _init_constants(self) -> None:
        """Joint addresses, limits, step sizes, home pose, object bookkeeping (identical in every world)."""
        cfg, m, dev = self.cfg, self.mjm, self.device
        assert m.actuator(0).name == ARM_JOINTS[0] and m.nu == N_ARM + N_HAND
        t = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=dev)
        jnt = m.actuator_trnid[:, 0]
        self.qadr, self.vadr = t(m.jnt_qposadr[jnt], torch.long), t(m.jnt_dofadr[jnt], torch.long)
        lo, hi = m.jnt_range[jnt].T.copy()
        home = np.concatenate([cfg.arm_home, np.zeros(N_HAND)])
        lo[:N_ARM] = np.maximum(lo[:N_ARM], home[:N_ARM] - np.asarray(cfg.arm_range))
        hi[:N_ARM] = np.minimum(hi[:N_ARM], home[:N_ARM] + np.asarray(cfg.arm_range))
        self.q_lo, self.q_hi, self.q_home = t(lo), t(hi), t(np.clip(home, lo, hi))
        self.q_scale = t(np.concatenate([cfg.arm_range, (hi[N_ARM:] - lo[N_ARM:]) / 2]))
        self.q_mid = t(np.concatenate([home[:N_ARM], (hi[N_ARM:] + lo[N_ARM:]) / 2]))
        self.step_size = t(np.repeat([cfg.arm_step, cfg.hand_step], [N_ARM, N_HAND]))
        self.lead = t(np.repeat(cfg.max_lead, [N_ARM, N_HAND]))
        n_obj = len(self.lib_ids)
        self.obj_body = t([m.body(f"obj{j}").id for j in range(n_obj)], torch.long)
        self.obj_qadr = t([m.jnt_qposadr[m.joint(f"obj{j}").id] for j in range(n_obj)], torch.long)
        self.obj_vadr = t([m.jnt_dofadr[m.joint(f"obj{j}").id] for j in range(n_obj)], torch.long)
        self.obj_weight = t(-m.body_subtreemass[self.obj_body.cpu().numpy()] * m.opt.gravity[2])  # N
        self.qpos0 = t(m.qpos0).clone()
        self.qpos0[self.qadr] = self.q_home
        for j in range(n_obj):
            self.qpos0[self.obj_qadr[j]:self.obj_qadr[j] + 7] = t([*park_position(j), 1, 0, 0, 0])
        self.hand_root, self.tcp_site = m.body(f"{HAND}right_hand_C_MC").id, m.site(f"{HAND}tcp").id
        self.flange_site = m.site(f"{HAND}flange").id
        self.sens_force = m.sensor_adr[m.sensor("flange_force").id]
        self.sens_torque = m.sensor_adr[m.sensor("flange_torque").id]
        self.hand_mass = float(m.body_subtreemass[self.hand_root])
        self.gravity = t(m.opt.gravity)
        self.rim_z = cfg.bin.floor_z + cfg.bin.wall_height
        self.max_steps = round(cfg.timeout / (cfg.timestep * cfg.substeps))
        d = mujoco.MjData(m)
        d.qpos[:] = self.qpos0.cpu().numpy()
        mujoco.mj_kinematics(m, d)
        self.tcp_home = t(d.site_xpos[self.tcp_site])

    # ---------------------------------------------------------------------------------------------- public API --
    def reset(self) -> dict:
        """Reset all worlds. Returns the batched first observation (see `step`)."""
        self._reset(torch.ones(self.n, dtype=torch.bool, device=self.device))
        return self._obs(self._read(), first=torch.ones_like(self.t, dtype=torch.bool))

    def step(self, action: torch.Tensor) -> tuple[dict, torch.Tensor, torch.Tensor, dict]:
        """Advance all worlds by one control step (substeps x timestep s).

        Args:
            action: (N, 28) in [-1, 1], joint-target increments (6 arm then 22 hand) in units of the per-joint step.
                Worlds with an override (`override`) use the override action instead.

        Returns:
            obs: "o" (N, D_o), "pts" (N, 5, 7) [pos_base m, normal_base, force N], "pts_valid" (N, 5) bool,
                "first" (N,) bool, "target" (N,) long library index of the target.
            reward: (N,) float32.
            done: (N,) bool; finished worlds are reset and their obs is the first of the next episode.
            info: privileged truth of the step just taken (before any auto-reset): "touched", "tip_obj",
                "touch_target", "conf_gt", "success", "wrong_pick", "timeout", "priv" (see `_labels`).
        """
        action = torch.where(self.override_mask[:, None], self.override_action, action.to(self.device)).clamp(-1, 1)
        q = self.sim.qpos[:, self.qadr]
        target = torch.clamp(self.q_target + action * self.step_size, self.q_lo, self.q_hi)
        self.q_target = torch.clamp(target, q - self.lead, q + self.lead)
        self.sim.set_ctrl(self.q_target)
        self.sim.step(self.cfg.substeps)
        self.t += 1
        r = self._read()
        info = self._labels(r)
        w = self.cfg.reward
        tz = self.sim.xpos[torch.arange(self.n, device=self.device), self.obj_body[self.active_target()], 2]
        reward = (w.success * info["success"] + w.wrong_pick * info["wrong_pick"]
                  + w.touch_target * info["touch_target"] + w.lift * (tz - self.target_z) + w.time
                  + w.action_rate * (action - self.prev_action).square().mean(1))
        self.target_z, self.prev_action = tz, action
        done = info["success"] | info["wrong_pick"] | info["timeout"]
        self._reset(done)  # masked: no host sync; the extra forward also refreshes the reset worlds' readings
        r = self._read()
        self.last = {**info, "reward": reward, "done": done}
        return self._obs(r, first=done), reward.float(), done, info

    def active_target(self) -> torch.Tensor:
        """(N,) long object body index j of each world's target."""
        return self.active.gather(1, self.target[:, None])[:, 0]

    def world_state(self, i: int) -> dict:
        """One world's state for display, as CPU numpy: qpos, tcp position, contact points, labels, objects."""
        r = self.reading
        out = {"qpos": self.sim.qpos[i], "tcp": self.sim.site_xpos[i, self.tcp_site], "pts": r["point_base"][i],
               "normals": r["normal_base"][i], "force": r["force"][i], "pts_valid": r["valid"][i],
               "active_lib": self.lib_ids[self.active[i]], "active_body": self.obj_body[self.active[i]],
               "target_slot": self.target[i], "target_lib": self.lib_ids[self.active_target()[i]], "t": self.t[i]}
        out |= {k: v[i] for k, v in self.last.items()}
        return {k: v.detach().cpu().numpy() for k, v in out.items()}

    def override(self, i: int, action: torch.Tensor | None) -> None:
        """Drive world i with `action` (28,) from outside (e.g. a browser viewer); None hands it back to the caller."""
        self.override_mask[i] = action is not None
        if action is not None:
            self.override_action[i] = torch.as_tensor(action, dtype=torch.float32, device=self.device)

    # ---------------------------------------------------------------------------------------------- internals --
    def _drop(self, mask: torch.Tensor) -> None:
        """Masked worlds: draw K distinct objects and place them at random poses low over the bin floor, one per
        bin quadrant (shuffled) and staggered in height, so they neither overlap nor land on the hand."""
        cfg, n, k, dev = self.cfg, self.n, self.cfg.n_slots, self.device
        keys = torch.rand(n, len(self.lib_ids), generator=self.gen, device=dev)
        self.active = torch.where(mask[:, None], keys.argsort(1)[:, :k], self.active)
        radius = self.clouds[self.lib_ids[self.active], :, :3].norm(dim=-1).amax(-1)  # (N, K)
        quarter = torch.tensor(cfg.bin.inner, device=dev) / 4
        quad = torch.rand(n, 4, generator=self.gen, device=dev).argsort(1)[:, :k]  # (N, K) quadrant of each slot
        sign = torch.stack([quad % 2, quad // 2], -1) * 2 - 1.0
        room = (quarter - radius[..., None]).clamp(min=0)
        jitter = (torch.rand(n, k, 2, generator=self.gen, device=dev) * 2 - 1) * room
        xy = torch.tensor(cfg.bin.center, device=dev) + sign * quarter + jitter
        z = cfg.bin.floor_z + radius + 0.005 + 0.02 * torch.arange(k, device=dev)
        quat = torch.nn.functional.normalize(torch.randn(n, k, 4, generator=self.gen, device=dev), dim=-1)
        self._set(mask, torch.cat([xy, z[..., None], quat], -1))

    def _set(self, mask: torch.Tensor, obj_pose: torch.Tensor) -> None:
        """Masked worlds: robot at home, active objects at `obj_pose` (N, K, 7), others parked; zero velocities."""
        qpos = self.qpos0.expand(self.n, -1).clone()
        idx = self.obj_qadr[self.active][..., None] + torch.arange(7, device=self.device)  # (N, K, 7)
        qpos.scatter_(1, idx.flatten(1), obj_pose.flatten(1))
        self.sim.set_state(mask, qpos, torch.zeros(self.n, self.mjm.nv, device=self.device))
        self.q_target = torch.where(mask[:, None], self.q_home, self.q_target)
        self.sim.set_ctrl(self.q_target)
        # parked objects float: an applied force cancels their weight (active objects get none)
        parked = torch.ones(self.n, len(self.lib_ids), dtype=torch.bool, device=self.device)
        parked.scatter_(1, self.active, False)
        xfrc = torch.zeros(self.n, self.mjm.nbody, 6, device=self.device)
        xfrc[:, self.obj_body, 2] = parked * self.obj_weight
        self.sim.set_xfrc(xfrc)
        self.obj_slot = torch.full((self.n, len(self.lib_ids)), -1, dtype=torch.long, device=self.device)
        self.obj_slot.scatter_(1, self.active, torch.arange(self.cfg.n_slots, device=self.device).expand(self.n, -1))

    def _obj_rows(self, field: torch.Tensor, adr: torch.Tensor, width: int) -> torch.Tensor:
        """Free-joint entries (N, K, width) of each world's active objects from qpos (width 7) or qvel (width 6)."""
        rows = torch.arange(self.n, device=self.device)[:, None, None]
        return field[rows, adr[self.active][..., None] + torch.arange(width, device=self.device)]

    def _build_bank(self, size: int) -> None:
        """Drop and settle all worlds at once, repeatedly, keeping states whose objects all rest inside the bin."""
        cfg, all_ = self.cfg, torch.ones(self.n, dtype=torch.bool, device=self.device)
        poses, actives, half = [], [], torch.tensor(cfg.bin.inner, device=self.device) / 2
        for _ in range(4 * -(-size // self.n)):  # at most 4x the rounds needed if every state were kept
            self._drop(all_)
            self.sim.step(round(cfg.settle_time / cfg.timestep))
            pose = self._obj_rows(self.sim.qpos, self.obj_qadr, 7)
            rel = (pose[..., :2] - torch.tensor(cfg.bin.center, device=self.device)).abs()
            ok = ((rel < half).all(-1) & (pose[..., 2] > cfg.bin.floor_z) & (pose[..., 2] < self.rim_z)).all(1)
            ok &= self._obj_rows(self.sim.qvel, self.obj_vadr, 6).flatten(1).abs().amax(1) < 0.2
            poses.append(pose[ok]), actives.append(self.active[ok])
            if sum(len(p) for p in poses) >= size:
                break
        self.bank_pose, self.bank_active = torch.cat(poses)[:size], torch.cat(actives)[:size]
        assert len(self.bank_pose) > 0, "no settled state stayed inside the bin"

    def _reset(self, mask: torch.Tensor) -> None:
        """Masked worlds start a new episode from a random bank state, with a random target slot (no host sync)."""
        pick = torch.randint(len(self.bank_pose), (self.n,), generator=self.gen, device=self.device)
        self.active = torch.where(mask[:, None], self.bank_active[pick], self.active)
        current = self._obj_rows(self.sim.qpos, self.obj_qadr, 7)
        self._set(mask, torch.where(mask[:, None, None], self.bank_pose[pick], current))
        tgt = torch.randint(self.cfg.n_slots, (self.n,), generator=self.gen, device=self.device)
        self.target = torch.where(mask, tgt, self.target)
        self.t = torch.where(mask, 0, self.t)
        self.prev_action = torch.where(mask[:, None], 0.0, self.prev_action)
        self.last_touch = torch.where(mask[:, None], -(10**9), self.last_touch)
        self.sim.forward()
        tz = self.sim.xpos[torch.arange(self.n, device=self.device), self.obj_body[self.active_target()], 2]
        self.target_z = torch.where(mask, tz, self.target_z)

    def _read(self) -> dict:
        """Tactile readings of all worlds (kept for `world_state`)."""
        self.reading = self.tactile.read(self.sim, self.obj_slot, self.cfg.n_slots)
        return self.reading

    def _labels(self, r: dict) -> dict:
        """Privileged simulator truth after a step (updates the coverage memory)."""
        cfg, sim, n, dev = self.cfg, self.sim, self.n, self.device
        rows = torch.arange(n, device=dev)
        body = self.obj_body[self.active]  # (N, K)
        pos, mat = sim.xpos[rows[:, None], body], sim.xmat[rows[:, None], body]  # object frames = cloud frames
        tgt = self.target
        # coverage memory: target surface points near a pad's contact point on the target, in the object frame
        on_tgt = r["valid"] & (r["slot_force"][rows, :, tgt] >= cfg.force_threshold)  # (N, 5)
        p_obj = torch.einsum("nij,nki->nkj", mat[rows, tgt], r["slot_point"][rows, :, tgt] - pos[rows, tgt, None])
        cloud = self.clouds[self.lib_ids[self.active_target()], :, :3]  # (N, P, 3)
        near = ((torch.cdist(p_obj, cloud) < cfg.coverage.radius) & on_tgt[..., None]).any(1)  # (N, P)
        self.last_touch = torch.where(near, self.t[:, None], self.last_touch)
        coverage = ((self.t[:, None] - self.last_touch) < self.history).float().mean(1) / cfg.coverage.full_at
        touch_now = on_tgt.any(1)
        per_slot = r["slot_force"].sum(1)  # (N, K)
        lib_active = self.lib_ids[self.active]
        lowest = torch.einsum("nkj,nkpj->nkp", mat[..., 2, :], self.clouds[lib_active, :, :3]).amin(-1)
        out = lowest + pos[..., 2] > self.rim_z + cfg.pull_height  # (N, K) pulled out of the bin
        is_tgt = torch.nn.functional.one_hot(tgt, cfg.n_slots).bool()
        tcp = sim.site_xpos[:, self.tcp_site]
        priv = torch.cat([(pos - tcp[:, None]) / 0.25, rot6d(mat), is_tgt[..., None].float()], -1)
        return {
            "touched": torch.where(per_slot.amax(1) > 0, lib_active.gather(1, per_slot.argmax(1, True))[:, 0], -1),
            "tip_obj": torch.where(r["slot"] >= 0, lib_active.gather(1, r["slot"].clamp(min=0)), -1),
            "touch_target": touch_now,
            "conf_gt": coverage.clamp(max=1.0) if cfg.label == "coverage" else touch_now.float(),
            "success": out[rows, tgt],
            "wrong_pick": (out & ~is_tgt).any(1),
            "timeout": self.t >= self.max_steps,
            "priv": priv.flatten(1),
        }

    def _obs(self, r: dict, first: torch.Tensor) -> dict:
        """Observation: the configured groups of o_t plus the contact points."""
        sim = self.sim
        q, qd, n_a = sim.qpos[:, self.qadr], sim.qvel[:, self.vadr], N_ARM
        rot = sim.site_xmat[:, self.flange_site]  # (N, 3, 3)
        mg = self.hand_mass * self.gravity  # payload compensation, as on the UR controller
        arm = sim.subtree_com[:, self.hand_root] - sim.site_xpos[:, self.flange_site]
        force = sim.sensordata[:, self.sens_force:self.sens_force + 3] + mg @ rot
        torque = sim.sensordata[:, self.sens_torque:self.sens_torque + 3] + torch.einsum(
            "nji,nj->ni", rot, torch.cross(arm, mg.expand_as(arm), dim=1))
        tcp_rot = sim.site_xmat[:, self.tcp_site]
        groups = {
            "arm_q": lambda: (q[:, :n_a] - self.q_mid[:n_a]) / self.q_scale[:n_a],
            "arm_qd": lambda: qd[:, :n_a] / 1.0,
            "hand_q": lambda: (q[:, n_a:] - self.q_mid[n_a:]) / self.q_scale[n_a:],
            "hand_qd": lambda: qd[:, n_a:] / 5.0,
            "q_target": lambda: (self.q_target - self.q_mid) / self.q_scale,
            "tcp_pose": lambda: torch.cat([(sim.site_xpos[:, self.tcp_site] - self.tcp_home) / 0.25, rot6d(tcp_rot)],
                                          1),
            "tcp_wrench": lambda: torch.cat([force / 20.0, torque / 2.0], 1),
            "tip_wrench": lambda: (r["wrench_tip"] / torch.tensor([5.0] * 3 + [0.05] * 3, device=q.device)).flatten(1),
            "tip_force": lambda: r["force"] / 5.0,
            "tip_contact": lambda: r["valid"].float(),
            "tip_point": lambda: r["point_tip"].flatten(1) / 0.02,
            "tip_normal": lambda: r["normal_tip"].flatten(1),
            "prev_action": lambda: self.prev_action,
        }
        return {
            "o": torch.cat([groups[g]() for g in self.cfg.obs_groups], 1).float(),
            "pts": torch.cat([r["point_base"], r["normal_base"], r["force"][..., None]], -1).float(),
            "pts_valid": r["valid"].clone(),
            "first": first.clone(),
            "target": self.lib_ids[self.active_target()],
        }
