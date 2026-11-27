"""Batched physics backends behind one small interface: `WarpSim` (MuJoCo Warp, GPU or CPU) and `CpuSim` (reference).

`BinEnv` only talks to this interface, so the whole env runs on the simulation device as torch tensors. Read-only
state is exposed as float32 tensors of shape (nworld, ...); writes go through `set_*`. One MjModel serves all worlds.
"""

from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
import torch

READ_FIELDS = ("qpos", "qvel", "xpos", "xmat", "geom_xpos", "site_xpos", "site_xmat", "subtree_com", "sensordata")


class CpuSim:
    """Local test and reference backend: N `MjData` of one `MjModel` (float64), stepped in a thread pool.

    Not for experiments (CPU, per-world Python). It mirrors `WarpSim` field by field so the env and the parity test
    run unchanged on both. Tensors are refreshed after `step` / `forward`.

    Args:
        model: the scene model (contacts must use elliptic cones, see `contacts`).
        nworld: number of worlds.
        n_threads: worker threads (`mj_step` releases the GIL for all its substeps).
    """

    def __init__(self, model: mujoco.MjModel, nworld: int, n_threads: int = 8, **_):
        assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
        self.model, self.nworld, self.device = model, nworld, torch.device("cpu")
        self.datas = [mujoco.MjData(model) for _ in range(nworld)]
        self.pool = ThreadPoolExecutor(min(n_threads, nworld))
        self.ctrl = torch.zeros(nworld, model.nu)
        self.xfrc_applied = torch.zeros(nworld, model.nbody, 6)
        self.forward()

    def set_state(self, mask: torch.Tensor, qpos: torch.Tensor, qvel: torch.Tensor) -> None:
        """Overwrite qpos (N, nq) and qvel (N, nv) of the worlds where `mask` (N,) is set; clear their solver state."""
        q, v = qpos.double().numpy(), qvel.double().numpy()
        for r in mask.nonzero()[:, 0].tolist():
            d = self.datas[r]
            d.qpos[:], d.qvel[:], d.qacc_warmstart[:] = q[r], v[r], 0.0

    def set_ctrl(self, ctrl: torch.Tensor) -> None:
        """Actuator controls (nworld, nu) held for the next steps."""
        self.ctrl[:] = ctrl

    def set_xfrc(self, xfrc: torch.Tensor) -> None:
        """Applied body wrenches (nworld, nbody, 6) [force N, torque N m], base frame, held for the next steps."""
        self.xfrc_applied[:] = xfrc

    def step(self, nstep: int) -> None:
        """Advance every world by `nstep` physics steps with the held controls and wrenches."""
        ctrl, xfrc = self.ctrl.double().numpy(), self.xfrc_applied.double().numpy()

        def run(i: int) -> None:
            d = self.datas[i]
            d.ctrl[:], d.xfrc_applied[:] = ctrl[i], xfrc[i]
            mujoco.mj_step(self.model, d, nstep=nstep)

        list(self.pool.map(run, range(self.nworld)))
        self._pull()

    def forward(self) -> None:
        """Recompute kinematics, contacts, constraint forces and sensors without advancing time."""
        for d in self.datas:
            mujoco.mj_forward(self.model, d)
        self._pull()

    def contacts(self) -> dict:
        """Flat contact list of all worlds (padded layout as in `WarpSim`).

        Returns:
            dict of tensors over contacts C: "world" (C,) long, "geom" (C, 2) long, "pos" (C, 3) base frame m,
            "frame" (C, 3, 3) rows = normal (geom0 -> geom1), tangents, "force" (C, 3) N in the contact frame (force of
            geom0 on geom1), "valid" (C,) bool.
        """
        rows = []
        for w, d in enumerate(self.datas):
            n = d.ncon
            f = np.zeros((n, 3))
            for c in range(n):  # elliptic cones: efc_force holds [normal, tangent1, tangent2] directly
                adr = d.contact.efc_address[c]
                f[c] = d.efc_force[adr:adr + 3] if adr >= 0 else 0.0
            rows.append((np.full(n, w), d.contact.geom[:n], d.contact.pos[:n], d.contact.frame[:n].reshape(n, 3, 3), f))
        world, geom, pos, frame, force = (np.concatenate(x) for x in zip(*rows))
        t = lambda x, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=dt)
        return {"world": t(world, torch.long), "geom": t(geom, torch.long), "pos": t(pos), "frame": t(frame),
                "force": t(force), "valid": torch.ones(len(world), dtype=torch.bool)}

    def health(self) -> dict:
        """Buffer use (CPU MuJoCo grows its buffers itself): largest contact and constraint counts of any world."""
        return {"overflow": 0, "ncon_max": max(d.ncon for d in self.datas), "nefc_max": max(d.nefc for d in self.datas)}

    def _pull(self) -> None:
        """Copy the read fields of all worlds into stacked float32 tensors."""
        for f in READ_FIELDS:
            arr = np.stack([getattr(d, f) for d in self.datas])
            if f.endswith("xmat"):
                arr = arr.reshape(*arr.shape[:-1], 3, 3)
            setattr(self, f, torch.as_tensor(arr, dtype=torch.float32))


class WarpSim:
    """Experiment backend: MuJoCo Warp, all worlds batched on one device ("cuda" or "cpu"; float32).

    Torch tensors are zero-copy views of the Warp arrays. On CUDA the substeps of one control tick run as one captured
    CUDA graph; on the CPU device they are plain kernel launches (slow, for local tests only).

    Args:
        model: the scene model.
        nworld: number of worlds.
        device: torch / warp device string, e.g. "cuda", "cuda:1", "cpu".
        nconmax: contacts allocated per world (pooled over all worlds: one world may use more).
        njmax: constraint rows allocated per world (hard per-world cap).
    """

    def __init__(self, model: mujoco.MjModel, nworld: int, device: str = "cuda", nconmax: int = 64,
                 njmax: int = 300, **_):
        import warp as wp

        wp.config.log_level = wp.LOG_WARNING  # no per-kernel load messages (set before mujoco_warp initialises warp)
        import mujoco_warp as mjw

        assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
        self.mjw, self.wp, self.model, self.nworld, self.njmax = mjw, wp, model, nworld, njmax
        self.wdevice = wp.get_device(device)
        self.device = torch.device(device)
        d = mujoco.MjData(model)
        mujoco.mj_forward(model, d)
        with wp.ScopedDevice(self.wdevice):
            self.mw = mjw.put_model(model)
            self.dw = mjw.put_data(model, d, nworld=nworld, nconmax=nconmax, njmax=njmax)
        t, dw = wp.to_torch, self.dw
        self.ctrl, self.xfrc_applied, self._qpos, self._qvel = t(dw.ctrl), t(dw.xfrc_applied), t(dw.qpos), t(dw.qvel)
        self._carry = [t(dw.qacc_warmstart), t(dw.cvel), t(dw.cdof_dot)]  # per-world arrays a reset must clear
        self._graphs = {}
        self.forward()  # compiles and loads the kernels before any graph capture

    def __getattr__(self, name: str) -> torch.Tensor:
        """Read fields (`READ_FIELDS`) as zero-copy torch views of the current Warp arrays."""
        if name in READ_FIELDS:
            return self.wp.to_torch(getattr(self.__dict__["dw"], name))
        raise AttributeError(name)

    def set_state(self, mask: torch.Tensor, qpos: torch.Tensor, qvel: torch.Tensor) -> None:
        """Overwrite qpos (N, nq) and qvel (N, nv) of the worlds where `mask` (N,) is set, without a host sync, and
        clear their carried-over solver state (warmstart, cvel, cdof_dot: else a world that once went NaN stays NaN,
        and a new episode would start from the old one's accelerations)."""
        m = mask[:, None]
        self._qpos.copy_(torch.where(m, qpos.to(self._qpos), self._qpos))
        self._qvel.copy_(torch.where(m, qvel.to(self._qvel), self._qvel))
        for x in self._carry:
            x.copy_(torch.where(mask.view(-1, *[1] * (x.dim() - 1)), 0.0, x))  # where, not *: NaN * 0 = NaN

    def set_ctrl(self, ctrl: torch.Tensor) -> None:
        """Actuator controls (nworld, nu) held for the next steps."""
        self.ctrl.copy_(ctrl)

    def set_xfrc(self, xfrc: torch.Tensor) -> None:
        """Applied body wrenches (nworld, nbody, 6) [force N, torque N m], base frame, held for the next steps."""
        self.xfrc_applied.copy_(xfrc)

    def step(self, nstep: int) -> None:
        """Advance every world by `nstep` physics steps (one CUDA graph on GPU)."""
        self._launch(nstep, lambda: [self.mjw.step(self.mw, self.dw) for _ in range(nstep)])

    def forward(self) -> None:
        """Recompute kinematics, contacts, constraint forces and sensors without advancing time."""
        self._launch("forward", lambda: self.mjw.forward(self.mw, self.dw))

    def _launch(self, key, fn) -> None:
        """Run `fn` (kernel launches only); on CUDA capture it once as a graph under `key` and replay that."""
        wp = self.wp
        with wp.ScopedDevice(self.wdevice):
            if not self.wdevice.is_cuda:
                fn()
                return
            if key not in self._graphs:
                with wp.ScopedCapture() as cap:
                    fn()
                self._graphs[key] = cap.graph
            wp.capture_launch(self._graphs[key])  # warp's stream is blocking: it orders with torch's default stream

    def contacts(self) -> dict:
        """Flat padded contact list of all worlds, no host sync (see `CpuSim.contacts`; "valid" masks padding)."""
        t, c = self.wp.to_torch, self.dw.contact
        nacon = t(self.dw.nacon)
        world, adr = t(c.worldid).long(), t(c.efc_address)[:, :3].long()
        valid = torch.arange(len(world), device=self.device) < nacon
        force = t(self.dw.efc.force)[world.clamp(min=0)[:, None], adr.clamp(min=0)]
        valid_force = valid[:, None] & (adr >= 0) & (adr < self.njmax)
        v = valid[:, None]
        return {"world": world.clamp(min=0), "geom": t(c.geom).long().clamp(min=0),
                "pos": torch.where(v, t(c.pos), 0.0), "frame": torch.where(v[..., None], t(c.frame), 0.0),
                "force": torch.where(valid_force, force, 0.0), "valid": valid}  # padding rows zeroed: may be stale

    def health(self) -> dict:
        """Buffer use after the last step (host sync: for checks and logs, not every tick)."""
        dw = self.dw
        return {"overflow": int((dw.overflow.numpy() != 0).sum()), "nacon": int(dw.nacon.numpy()[0]),
                "nacon_cap": int(dw.contact.dist.shape[0]), "nefc_max": int(dw.nefc.numpy().max()), "njmax": self.njmax}


def make_sim(backend: str, model: mujoco.MjModel, nworld: int, device: str, **kw):
    """Create the configured backend: "warp" (experiments) or "cpu" (reference / local tests)."""
    return WarpSim(model, nworld, device, **kw) if backend == "warp" else CpuSim(model, nworld, **kw)
