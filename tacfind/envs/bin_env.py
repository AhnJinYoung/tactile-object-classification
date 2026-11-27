"""Batched table environment: blind touch search for a target object among 3-8 on a table, then lift and hold it.

(Module and class keep the historical name `bin_env.BinEnv`; there is no bin.)

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
from tacfind.envs.randomize import Randomizer
from tacfind.envs.scene import ARM_JOINTS, HAND, build_scene, park_position
from tacfind.envs.sim import make_sim
from tacfind.envs.tactile import N_TIPS, Tactile
from tacfind.spec import EnvSpec

REPO = Path(__file__).resolve().parents[2]
N_ARM, N_HAND = 6, 22
GROUP_SIZES = {"arm_q": 6, "arm_qd": 6, "hand_q": 22, "hand_qd": 22, "q_target": 28, "tcp_pose": 9, "tcp_wrench": 6,
               "tip_wrench": 30, "tip_force": 5, "tip_contact": 5, "tip_point": 15, "tip_normal": 15, "prev_action": 28}
GRID = (3, 4)  # cells of the object zone (x, y): one object per cell, so up to 12 objects never overlap
PRIV_PER_SLOT = 10  # object position relative to the tcp (3), 6-D rotation (6), is-target flag (1)
# priv is padded to cfg.n_slots slots: empty slots are all zeros (their is-target flag included)


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
    """N parallel table worlds in one batched simulation.

    Each world holds n_active (3-8) objects of the split in slots 0..n_active-1 of K = min(n_slots, split size) slots;
    the other slots are empty (`slot_valid` False) and their objects are parked.

    Args:
        cfg: the `env` config node (configs/env/bin.yaml).
        n_envs: number N of worlds.
        history: H, steps of contact memory behind the coverage label.
        device: simulation device ("cuda", "cuda:1", "cpu"); the cpu backend always runs on "cpu".
        seed: seeds the settled-state bank, every reset and the domain randomisation.
        split: "train" or "heldout" library objects.
    """

    def __init__(self, cfg, n_envs: int, history: int, device: str = "cuda", seed: int = 0, split: str = "train"):
        self.cfg = cfg = plain(cfg)
        self.n, self.history = n_envs, history
        self.root = (REPO / cfg.assets).resolve()
        lib = build_library(cfg.objects, self.root)
        self.lib_ids = torch.as_tensor(split_indices(cfg.objects, split))  # library index of object body j
        self.split_objects = [lib[i] for i in self.lib_ids.tolist()]
        self.mjm = build_scene(cfg, self.root, self.split_objects)
        self.sim = make_sim(cfg.sim, self.mjm, n_envs, device, nconmax=cfg.nconmax, njmax=cfg.njmax,
                            solver_iterations=cfg.solver_iterations, nativeccd=cfg.warp_nativeccd,
                            n_threads=cfg.n_threads)
        dev = self.device = self.sim.device
        self.clouds = torch.as_tensor(np.stack([o.cloud for o in lib]), device=dev)  # (n_lib, N_pts, 6)
        self.lib_ids = self.lib_ids.to(dev)
        self.k = min(cfg.n_slots, len(self.lib_ids))  # slots in use (a small held-out split may have fewer objects)
        self.tactile = Tactile(self.mjm, cfg.force_threshold, dev, cfg.pad_only, cfg.pad_margin)
        self.gen = torch.Generator(dev).manual_seed(seed)
        self._init_constants()
        self.spec = EnvSpec(obs_dim=sum(GROUP_SIZES[g] for g in cfg.obs_groups), act_dim=N_ARM + N_HAND,
                            priv_dim=PRIV_PER_SLOT * cfg.n_slots, n_tips=N_TIPS, pt_dim=7)
        z = lambda *s, dt=torch.float32: torch.zeros(n_envs, *s, dtype=dt, device=dev)
        self.q_target, self.prev_action, self.t = z(N_ARM + N_HAND), z(N_ARM + N_HAND), z(dt=torch.long)
        self.active, self.target = z(self.k, dt=torch.long), z(dt=torch.long)  # object bodies j per slot; target slot
        self.slot_valid, self.off_table = z(self.k, dt=torch.bool), z(self.k, dt=torch.bool)
        self.held = z(self.k, dt=torch.long)  # consecutive steps each object has been lifted
        self.target_z, self.last_touch = z(), z(self.clouds.shape[1], dt=torch.long)
        self.override_mask, self.override_action = z(dt=torch.bool), z(N_ARM + N_HAND)
        self.xfrc_park, self.xfrc_extra = z(self.mjm.nbody, 6), z(self.mjm.nbody, 6)  # parking + `perturb` wrenches
        self.randomizer = Randomizer(self)
        self._build_bank(cfg.bank_size)
        self.last, self._mjm_textured = {}, None

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
        self.obj_weight = t(-m.body_subtreemass[self.obj_body.cpu().numpy()] * m.opt.gravity[2])  # nominal, N
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
        self.table_z = cfg.table.top_z
        self.max_steps = round(cfg.timeout / (cfg.timestep * cfg.substeps))
        self.hold_steps = max(1, round(cfg.hold_time / (cfg.timestep * cfg.substeps)))
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
                "touch_target", "conf_gt", "success", "wrong_pick", "dropped", "target_dropped", "timeout",
                "n_objects", "priv" (see `_labels`).
        """
        action = torch.where(self.override_mask[:, None], self.override_action, action.to(self.device)).clamp(-1, 1)
        q = self.sim.qpos[:, self.qadr]
        target = torch.clamp(self.q_target + action * self.step_size, self.q_lo, self.q_hi)
        self.q_target = torch.clamp(target, q - self.lead, q + self.lead)
        self.sim.set_ctrl(self.q_target)
        self.sim.set_xfrc(self.xfrc_park + self.xfrc_extra)
        self.sim.step(self.cfg.substeps)
        self.sim.forward()  # contacts at the post-step state, as the reset path's forward: labels match the obs
        self.t += 1
        r = self._read()
        info, n_dropped = self._labels(r)
        w = self.cfg.reward
        tz = self.sim.xpos[torch.arange(self.n, device=self.device), self.obj_body[self.active_target()], 2]
        reward = (w.success * info["success"] + w.wrong_pick * info["wrong_pick"] + w.dropped * n_dropped
                  + w.touch_target * info["touch_target"] + w.lift * (tz - self.target_z) + w.time
                  + w.action_rate * (action - self.prev_action).square().mean(1))
        self.target_z, self.prev_action = tz, action
        done = info["success"] | info["wrong_pick"] | info["target_dropped"] | info["timeout"]
        self._reset(done)  # masked: no host sync; the extra forward also refreshes the reset worlds' readings
        r = self._read()
        self.last = {**info, "reward": reward, "done": done}
        return self._obs(r, first=done), reward.float(), done, info

    def active_target(self) -> torch.Tensor:
        """(N,) long object body index j of each world's target."""
        return self.active.gather(1, self.target[:, None])[:, 0]

    def world_state(self, i: int) -> dict:
        """One world's state for display, as CPU numpy: qpos, qvel, q_target, tcp position, contact points, labels,
        objects ("active_*" over the K slots, "active_valid" marks the filled ones)."""
        r = self.reading
        out = {"qpos": self.sim.qpos[i], "qvel": self.sim.qvel[i], "q_target": self.q_target[i],
               "tcp": self.sim.site_xpos[i, self.tcp_site], "pts": r["point_base"][i],
               "normals": r["normal_base"][i], "force": r["force"][i], "pts_valid": r["valid"][i],
               "active_lib": self.lib_ids[self.active[i]], "active_body": self.obj_body[self.active[i]],
               "active_valid": self.slot_valid[i], "target_slot": self.target[i],
               "target_lib": self.lib_ids[self.active_target()[i]], "t": self.t[i]}
        out |= {k: v[i] for k, v in self.last.items()}
        return {k: v.detach().cpu().numpy() for k, v in out.items()}

    def override(self, i: int, action: torch.Tensor | None) -> None:
        """Drive world i with `action` (28,) from outside (e.g. a browser viewer); None hands it back to the caller."""
        self.override_mask[i] = action is not None
        if action is not None:
            self.override_action[i] = torch.as_tensor(action, dtype=torch.float32, device=self.device)

    def reset_worlds(self, mask: torch.Tensor) -> None:
        """Start a new episode in the worlds where `mask` (N,) is set and refresh the readings (`world_state`). The
        caller's last obs is stale for those worlds; set its "first" entries for them if an actor keeps state."""
        self._reset(mask.to(self.device))
        self._read()

    def perturb(self, i: int, wrench: torch.Tensor | None) -> None:
        """Add an external wrench (nbody, 6) [force N, torque N m; base frame, applied at each body's centre of mass
        as `mjData.xfrc_applied`] to world i on top of the parking wrench, from the next step until cleared with None
        (e.g. pushes from a browser viewer)."""
        self.xfrc_extra[i] = 0.0 if wrench is None else torch.as_tensor(wrench, dtype=torch.float32, device=self.device)

    def textured_model(self) -> mujoco.MjModel:
        """Render-only copy of the model with the YCB textures (same bodies and joints: a physics qpos fits it)."""
        if self._mjm_textured is None:
            self._mjm_textured = build_scene(self.cfg, self.root, self.split_objects, textured=True)
        return self._mjm_textured

    # ---------------------------------------------------------------------------------------------- internals --
    def _drop(self, mask: torch.Tensor) -> None:
        """Masked worlds: draw n_active distinct objects and place them at random poses just above the table, one per
        cell of a grid over the zone (cells shuffled), so they do not overlap."""
        cfg, n, k, dev = self.cfg, self.n, self.k, self.device
        keys = torch.rand(n, len(self.lib_ids), generator=self.gen, device=dev)
        self.active = torch.where(mask[:, None], keys.argsort(1)[:, :k], self.active)
        lo, hi = min(cfg.n_active[0], k), min(cfg.n_active[1], k)
        n_act = torch.randint(lo, hi + 1, (n,), generator=self.gen, device=dev)
        self.slot_valid = torch.where(mask[:, None], torch.arange(k, device=dev) < n_act[:, None], self.slot_valid)
        size, ncell = torch.tensor(cfg.zone.size, device=dev), torch.tensor(GRID, device=dev)
        cell = torch.rand(n, int(ncell.prod()), generator=self.gen, device=dev).argsort(1)[:, :k]  # (N, K)
        cxy = torch.stack([cell % ncell[0], cell // ncell[0]], -1) + 0.5  # cell centres in cell units
        radius = self.clouds[self.lib_ids[self.active], :, :3].norm(dim=-1).amax(-1)  # (N, K)
        room = (size / ncell / 2 - radius[..., None]).clamp(min=0)
        xy = (torch.tensor(cfg.zone.center, device=dev) - size / 2 + cxy * size / ncell
              + (torch.rand(n, k, 2, generator=self.gen, device=dev) * 2 - 1) * room)
        z = self.table_z + radius + 0.005
        quat = torch.nn.functional.normalize(torch.randn(n, k, 4, generator=self.gen, device=dev), dim=-1)
        self._set(mask, torch.cat([xy, z[..., None], quat], -1))

    def _set(self, mask: torch.Tensor, obj_pose: torch.Tensor) -> None:
        """Masked worlds: robot at home, valid slots' objects at `obj_pose` (N, K, 7), all others parked."""
        park = self.qpos0[self.obj_qadr[self.active][..., None] + torch.arange(7, device=self.device)]
        obj_pose = torch.where(self.slot_valid[..., None], obj_pose, park)
        qpos = self.qpos0.expand(self.n, -1).clone()
        idx = self.obj_qadr[self.active][..., None] + torch.arange(7, device=self.device)  # (N, K, 7)
        qpos.scatter_(1, idx.flatten(1), obj_pose.flatten(1))
        self.sim.set_state(mask, qpos, torch.zeros(self.n, self.mjm.nv, device=self.device))
        self.q_target = torch.where(mask[:, None], self.q_home, self.q_target)
        self.sim.set_ctrl(self.q_target)
        self.randomizer.apply(mask)
        # parked objects float: an applied force cancels their (randomised) weight; objects on the table get none
        on_table = torch.zeros(self.n, len(self.lib_ids), dtype=torch.bool, device=self.device)
        on_table.scatter_(1, self.active, self.slot_valid)
        weight = -self.randomizer.mass[:, self.obj_body] * self.gravity[2]
        self.xfrc_park = torch.zeros(self.n, self.mjm.nbody, 6, device=self.device)
        self.xfrc_park[:, self.obj_body, 2] = ~on_table * weight
        self.sim.set_xfrc(self.xfrc_park + self.xfrc_extra)
        slot = torch.where(self.slot_valid, torch.arange(self.k, device=self.device), -1)
        self.obj_slot = torch.full((self.n, len(self.lib_ids)), -1, dtype=torch.long, device=self.device)
        self.obj_slot.scatter_(1, self.active, slot)

    def _obj_rows(self, field: torch.Tensor, adr: torch.Tensor, width: int) -> torch.Tensor:
        """Free-joint entries (N, K, width) of each world's slot objects from qpos (width 7) or qvel (width 6)."""
        rows = torch.arange(self.n, device=self.device)[:, None, None]
        return field[rows, adr[self.active][..., None] + torch.arange(width, device=self.device)]

    def _build_bank(self, size: int) -> None:
        """Drop and settle all worlds at once, repeatedly, keeping states whose objects all rest on the table."""
        cfg, all_ = self.cfg, torch.ones(self.n, dtype=torch.bool, device=self.device)
        poses, actives, valids = [], [], []
        for _ in range(4 * -(-size // self.n)):  # at most 4x the rounds needed if every state were kept
            self._drop(all_)
            self.sim.step(round(cfg.settle_time / cfg.timestep))
            pose, vel = self._obj_rows(self.sim.qpos, self.obj_qadr, 7), self._obj_rows(self.sim.qvel, self.obj_vadr, 6)
            rest = (pose[..., 2] > self.table_z) & (vel.abs().amax(-1) < 0.2)
            ok = (rest | ~self.slot_valid).all(1)
            poses.append(pose[ok]), actives.append(self.active[ok]), valids.append(self.slot_valid[ok])
            if sum(len(p) for p in poses) >= size:
                break
        self.bank_pose, self.bank_active = torch.cat(poses)[:size], torch.cat(actives)[:size]
        self.bank_valid = torch.cat(valids)[:size]
        assert len(self.bank_pose) > 0, "no settled state kept all objects on the table"

    def _reset(self, mask: torch.Tensor) -> None:
        """Masked worlds start a new episode from a random bank state, with a random target among the filled slots,
        and new randomised parameters (no host sync)."""
        pick = torch.randint(len(self.bank_pose), (self.n,), generator=self.gen, device=self.device)
        current = self._obj_rows(self.sim.qpos, self.obj_qadr, 7)
        self.active = torch.where(mask[:, None], self.bank_active[pick], self.active)
        self.slot_valid = torch.where(mask[:, None], self.bank_valid[pick], self.slot_valid)
        self._set(mask, torch.where(mask[:, None, None], self.bank_pose[pick], current))
        n_act = self.slot_valid.sum(1)
        tgt = (torch.rand(self.n, generator=self.gen, device=self.device) * n_act).long().clamp(max=self.k - 1)
        self.target = torch.where(mask, tgt, self.target)
        self.t = torch.where(mask, 0, self.t)
        self.held = torch.where(mask[:, None], 0, self.held)
        self.off_table = torch.where(mask[:, None], False, self.off_table)
        self.prev_action = torch.where(mask[:, None], 0.0, self.prev_action)
        self.last_touch = torch.where(mask[:, None], -(10**9), self.last_touch)
        self.sim.forward()
        tz = self.sim.xpos[torch.arange(self.n, device=self.device), self.obj_body[self.active_target()], 2]
        self.target_z = torch.where(mask, tz, self.target_z)

    def _read(self) -> dict:
        """Tactile readings of all worlds (kept for `world_state`), with the randomised threshold and noise."""
        rnd = self.cfg.randomize
        self.reading = self.tactile.read(self.sim, self.obj_slot, self.k, self.randomizer.threshold,
                                         rnd.force_noise if rnd.enabled else 0.0, self.gen)
        return self.reading

    def _labels(self, r: dict) -> tuple[dict, torch.Tensor]:
        """Privileged simulator truth after a step (updates the coverage, hold and off-table memories).

        Returns:
            info dict (see `step`) and the number of objects that left the table this step (N,).
        """
        cfg, sim, n, dev = self.cfg, self.sim, self.n, self.device
        rows, valid, tgt = torch.arange(n, device=dev), self.slot_valid, self.target
        body = self.obj_body[self.active]  # (N, K)
        pos, mat = sim.xpos[rows[:, None], body], sim.xmat[rows[:, None], body]  # object frames = cloud frames
        # coverage memory: target surface points near a pad's contact point on the target, in the object frame
        on_tgt = r["valid"] & (r["slot_force"][rows, :, tgt] >= self.randomizer.threshold[:, None])  # (N, 5)
        p_obj = torch.einsum("nij,nki->nkj", mat[rows, tgt], r["slot_point"][rows, :, tgt] - pos[rows, tgt, None])
        cloud = self.clouds[self.lib_ids[self.active_target()], :, :3]  # (N, P, 3)
        near = ((torch.cdist(p_obj, cloud) < cfg.coverage.radius) & on_tgt[..., None]).any(1)  # (N, P)
        self.last_touch = torch.where(near, self.t[:, None], self.last_touch)
        coverage = ((self.t[:, None] - self.last_touch) < self.history).float().mean(1) / cfg.coverage.full_at
        touch_now = on_tgt.any(1)
        per_slot = r["slot_force"].sum(1)  # (N, K)
        lib_active = self.lib_ids[self.active]
        lowest = torch.einsum("nkj,nkpj->nkp", mat[..., 2, :], self.clouds[lib_active, :, :3]).amin(-1) + pos[..., 2]
        lifted = valid & (lowest > self.table_z + cfg.lift_height)
        self.held = torch.where(lifted, self.held + 1, 0)
        held = self.held >= self.hold_steps
        off = valid & (pos[..., 2] < self.table_z - cfg.drop_margin)
        new_off = off & ~self.off_table
        self.off_table |= off
        is_tgt = torch.nn.functional.one_hot(tgt, self.k).bool()
        tcp = sim.site_xpos[:, self.tcp_site]
        priv = torch.cat([(pos - tcp[:, None]) / 0.25, rot6d(mat), is_tgt[..., None].float()], -1) * valid[..., None]
        priv = torch.nn.functional.pad(priv, (0, 0, 0, cfg.n_slots - self.k))  # (N, n_slots, 10)
        info = {
            "touched": torch.where(per_slot.amax(1) > 0, lib_active.gather(1, per_slot.argmax(1, True))[:, 0], -1),
            "tip_obj": torch.where(r["slot"] >= 0, lib_active.gather(1, r["slot"].clamp(min=0)), -1),
            "touch_target": touch_now,
            "conf_gt": coverage.clamp(max=1.0) if cfg.label == "coverage" else touch_now.float(),
            "success": held[rows, tgt],
            "wrong_pick": (held & ~is_tgt).any(1),
            "dropped": new_off.any(1),
            "target_dropped": off[rows, tgt],
            "timeout": self.t >= self.max_steps,
            "n_objects": valid.sum(1),
            "priv": priv.flatten(1),
        }
        return info, new_off.sum(1).float()

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
