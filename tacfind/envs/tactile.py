"""Batched contact-based fingertip tactile sensing for the 5 Sharpa Wave pads (torch, on the simulation device).

The real sensor outputs a 6-D wrench (`F6`) and a 240x240 deformation map from which contact position and normal are
derived. Simulating the map (e.g. by ray casting the elastomer) is out of scope for now: the force-weighted MuJoCo
contact point and the touched object's outward surface normal stand in for the map-derived quantities.

Each fingertip sensor collects the contacts of its whole distal phalanx (elastomer pad and phalanx collision hull):
the hull encloses all but ~1 mm of the pad, so objects touching the fingertip mostly hit both geoms and the stiffer hull
takes most of the force. The real sensor only senses on the pad, so with `pad_only` a contact counts only if its point
lies within the pad's bounding box (distal-phalanx frame, grown by `pad_margin`): touches on the nail side and the
phalanx base are dropped. Moments and frames refer to the pad centre.

Why contact arrays and not MuJoCo contact sensors (which MuJoCo Warp does support): a sensor reports neither which
object it touched (needed for labels) nor, with `reduce="netforce"`, a surface normal; one sensor per (pad, object)
pair would mean 5 x n_objects sensors evaluated at every substep. Scattering the contact list once per control step
gives everything from one pass and works identically on both backends (`sim.contacts()`).
"""

import mujoco
import numpy as np
import torch

from tacfind.envs.scene import FINGERS, HAND

N_TIPS = len(FINGERS)


class Tactile:
    """Aggregates pad-object contacts of all worlds.

    Args:
        model: the scene model; objects are the bodies `obj{j}`.
        threshold: minimum pad force magnitude that counts as contact, N.
        device: simulation device.
        pad_only: keep only contacts within the pad's bounding box.
        pad_margin: growth of that box, m.
    """

    def __init__(self, model: mujoco.MjModel, threshold: float, device: torch.device, pad_only: bool = True,
                 pad_margin: float = 0.002):
        self.threshold, self.pad_only = threshold, pad_only
        tip_of_geom, obj_of_geom = torch.full((model.ngeom,), -1), torch.full((model.ngeom,), -1)
        self.tip_body = torch.tensor([model.body(f"{HAND}right_{f}_DP").id for f in FINGERS])
        for g in range(model.ngeom):
            body, name = model.geom_bodyid[g], model.geom(g).name
            if body in self.tip_body and model.geom_contype[g]:  # elastomer pad and phalanx hull: see the note above
                tip_of_geom[g] = int((self.tip_body == body).nonzero())
            if model.body(body).name.startswith("obj"):
                obj_of_geom[g] = int(model.body(body).name[3:])
        self.pad_geom = torch.tensor([model.geom(f"{HAND}right_{f}_elastomer").id for f in FINGERS], device=device)
        self.tip_of_geom, self.obj_of_geom = tip_of_geom.to(device), obj_of_geom.to(device)
        self.tip_body = self.tip_body.to(device)
        box = []  # (5, 2, 3) pad bounding box in the distal-phalanx frame
        for g in self.pad_geom.tolist():
            mesh = model.geom_dataid[g]
            v = model.mesh_vert[model.mesh_vertadr[mesh]:][:model.mesh_vertnum[mesh]]
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, model.geom_quat[g])
            v = v @ rot.reshape(3, 3).T + model.geom_pos[g]
            box.append([v.min(0) - pad_margin, v.max(0) + pad_margin])
        self.pad_box = torch.tensor(np.array(box), dtype=torch.float32, device=device)

    def read(self, sim, obj_slot: torch.Tensor, n_slots: int, threshold: torch.Tensor | None = None,
             noise: float = 0.0, gen: torch.Generator | None = None) -> dict:
        """Current per-pad readings of all worlds.

        Args:
            sim: the backend (after `step` or `forward`).
            obj_slot: (N, n_objects) long, slot of each object body in each world, -1 if parked.
            n_slots: K.
            threshold: (N,) per-world contact threshold, N (None: the constructor's).
            noise: std of Gaussian noise added to each pad force component, N.
            gen: random generator of the noise.

        Returns:
            dict (rows = fingers thumb..pinky): "wrench_tip" / "wrench_base" (N, 5, 6) pad force N and moment about the
            pad centre N m; "point_tip" / "point_base" (N, 5, 3) force-weighted contact point m (tip frame relative to
            the pad centre); "normal_tip" / "normal_base" (N, 5, 3) outward unit normal of the touched object; "force"
            (N, 5) N; "valid" (N, 5) bool (force >= threshold); "slot" (N, 5) long (-1 if invalid); "slot_force"
            (N, 5, K) N; "slot_point" (N, 5, K, 3) force-weighted contact point per object, base frame m.
            Invalid pads read zero.
        """
        c, n = sim.contacts(), sim.nworld
        tip, obj = self.tip_of_geom[c["geom"]], self.obj_of_geom[c["geom"]]  # (C, 2)
        pad_first = (tip[:, 0] >= 0) & (obj[:, 1] >= 0)
        sel = c["valid"] & (pad_first | ((tip[:, 1] >= 0) & (obj[:, 0] >= 0)))
        i = torch.where(pad_first, tip[:, 0], tip[:, 1]).clamp(min=0)
        j = torch.where(pad_first, obj[:, 1], obj[:, 0]).clamp(min=0)
        slot = obj_slot[c["world"], j]
        if self.pad_only:  # contact point in the distal-phalanx frame must lie in the pad box
            body = self.tip_body[i]
            local = torch.einsum("cji,cj->ci", sim.xmat[c["world"], body], c["pos"] - sim.xpos[c["world"], body])
            sel &= ((local >= self.pad_box[i, 0]) & (local <= self.pad_box[i, 1])).all(1)
        sel &= slot >= 0
        slot = slot.clamp(min=0)
        sign = torch.where(pad_first, -1.0, 1.0)[:, None] * sel[:, None]  # +: the pad is geom1; 0: not a pad contact
        f_base = sign * torch.einsum("cji,cj->ci", c["frame"], c["force"])  # force on the pad, base frame
        normal = sign * c["frame"][:, 0]  # contact normal points from geom0 to geom1: here away from the object
        fn = c["force"][:, 0].clamp(min=0) * sel
        origin = sim.geom_xpos[:, self.pad_geom]  # (N, 5, 3) pad centres
        pad = c["world"] * N_TIPS + i
        moment = torch.cross(c["pos"] - origin.reshape(-1, 3)[pad], f_base, dim=1)
        scat = lambda v, idx, size: torch.zeros(size, *v.shape[1:], device=v.device).index_add_(0, idx, v)
        force = scat(f_base, pad, n * N_TIPS).view(n, N_TIPS, 3)
        moment = scat(moment, pad, n * N_TIPS).view(n, N_TIPS, 3)
        if noise > 0:  # sensor noise on every pad (moment noise: force noise x 0.01 m)
            force = force + noise * torch.randn(force.shape, generator=gen, device=force.device)
            moment = moment + 0.01 * noise * torch.randn(moment.shape, generator=gen, device=force.device)
        acc = scat(torch.cat([fn[:, None], fn[:, None] * c["pos"], fn[:, None] * normal], 1), pad, n * N_TIPS)
        acc = acc.view(n, N_TIPS, 7)
        pair = pad * n_slots + slot
        per_pair = torch.cat([fn[:, None], fn[:, None] * c["pos"], f_base.norm(dim=1, keepdim=True)], 1)
        acc_slot = scat(per_pair, pair, n * N_TIPS * n_slots).view(n, N_TIPS, n_slots, 5)
        mag = force.norm(dim=-1)
        thr = self.threshold if threshold is None else threshold[:, None]
        valid = (mag >= thr) & (acc[..., 0] > 0)  # a reading needs a real contact, not noise alone
        v3 = valid[..., None]
        point_base = torch.where(v3, acc[..., 1:4] / acc[..., :1].clamp(min=1e-9), 0.0)
        normal_base = torch.where(v3, torch.nn.functional.normalize(acc[..., 4:7], dim=-1), 0.0)
        slot_force = acc_slot[..., 4] * valid[..., None]
        rot = sim.xmat[:, self.tip_body]  # (N, 5, 3, 3) tip frame axes = distal phalanx body axes
        to_tip = lambda v: torch.einsum("nkji,nkj->nki", rot, v)  # R^T v
        return {
            "wrench_tip": torch.cat([to_tip(force), to_tip(moment)], -1),
            "wrench_base": torch.cat([force, moment], -1),
            "point_tip": torch.where(v3, to_tip(point_base - origin), 0.0),
            "point_base": point_base,
            "normal_tip": to_tip(normal_base),
            "normal_base": normal_base,
            "force": mag,
            "valid": valid,
            "slot": torch.where(valid, slot_force.argmax(-1), -1),
            "slot_force": slot_force,
            "slot_point": acc_slot[..., 1:4] / acc_slot[..., :1].clamp(min=1e-9),
        }
