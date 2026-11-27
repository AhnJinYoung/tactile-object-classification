"""Domain randomisation: per-world object mass, centre of mass, friction and tactile threshold, drawn at every reset.

Sits under `BinEnv._reset`. The physical parameters go to the backend as per-world model fields (`sim.set_model`;
MuJoCo Warp batches them with `put_model(batch_sizes)`); the tactile noise and threshold are applied in `tactile.py`.
"""

import torch


class Randomizer:
    """Draws and applies per-world parameters around the model's nominal values.

    Args:
        env: the environment (uses its model, sim, object bodies, generator and `cfg.randomize`).
    """

    def __init__(self, env):
        self.env, self.cfg, m, dev = env, env.cfg.randomize, env.mjm, env.device
        t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=dev)
        self.mass0, self.inertia0, self.ipos0 = t(m.body_mass), t(m.body_inertia), t(m.body_ipos)
        self.friction0, self.subtree0 = t(m.geom_friction), t(m.body_subtreemass)
        body = env.obj_body
        self.obj_geom = torch.isin(torch.as_tensor(m.geom_bodyid, device=dev), body)  # (ngeom,) object geoms
        self.geom_obj = torch.searchsorted(body, torch.as_tensor(m.geom_bodyid, device=dev).clamp(max=int(body.max())))
        n = env.n
        self.mass = self.mass0.expand(n, -1).clone()  # (N, nbody) current masses (parking weights use them)
        self.threshold = torch.full((n,), env.cfg.force_threshold, device=dev)  # (N,) tactile threshold, N

    def apply(self, mask: torch.Tensor) -> None:
        """Draw new parameters for the worlds in `mask` (N,) and write them to the backend."""
        if not self.cfg.enabled:
            return
        env, c, n, k = self.env, self.cfg, self.env.n, len(self.env.obj_body)
        u = lambda *s: torch.rand(n, *s, generator=env.gen, device=env.device)
        body = env.obj_body
        scale = c.mass_scale[0] + (c.mass_scale[1] - c.mass_scale[0]) * u(k)  # (N, K_obj) per object body
        mass, inertia, ipos = (x.expand(n, *x.shape).clone() for x in (self.mass0, self.inertia0, self.ipos0))
        mass[:, body] = self.mass0[body] * scale
        inertia[:, body] = self.inertia0[body] * scale[..., None]  # same shape, scaled mass
        ipos[:, body] = self.ipos0[body] + (u(k, 3) * 2 - 1) * c.com_offset
        fr = c.friction_scale[0] + (c.friction_scale[1] - c.friction_scale[0]) * u(k)
        friction = self.friction0.expand(n, -1, -1).clone()
        friction[:, self.obj_geom, 0] *= fr[:, self.geom_obj[self.obj_geom]]  # sliding friction of object geoms
        subtree = self.subtree0.expand(n, -1).clone()
        subtree[:, body] = mass[:, body]  # objects are leaf bodies
        for name, value in (("body_mass", mass), ("body_subtreemass", subtree), ("body_inertia", inertia),
                            ("body_ipos", ipos), ("geom_friction", friction)):
            env.sim.set_model(mask, name, value)
        self.mass = torch.where(mask[:, None], mass, self.mass)
        thr = c.force_threshold[0] + (c.force_threshold[1] - c.force_threshold[0]) * u()
        self.threshold = torch.where(mask, thr, self.threshold)
