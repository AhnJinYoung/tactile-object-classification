"""Object library: surface clouds lie on the simulated surface with unit outward normals, for every shape family."""

import mujoco
import numpy as np
import pytest
from omegaconf import OmegaConf

from tacfind.envs.objects import FAMILIES, build_library, inside, split_indices
from tacfind.envs.scene import add_object

REPO_CFG = "configs/env/bin.yaml"


@pytest.fixture(scope="module")
def lib():
    cfg = OmegaConf.load(REPO_CFG).objects
    cfg.n_objects = 2 * len(FAMILIES)  # every family twice
    return cfg, build_library(cfg)


@pytest.mark.parametrize("family", FAMILIES)
def test_cloud_on_surface_with_outward_normals(lib, family):
    cfg, objs = lib
    for obj in [o for o in objs if o.family == family]:
        cloud = obj.cloud.astype(np.float64)
        assert cloud.shape == (cfg.n_points, 6) and obj.cloud.dtype == np.float32
        pts, nrm = cloud[:, :3], cloud[:, 3:]
        assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-5)
        eps = 1e-4  # small: a step inward near a sharp (60 deg prism) edge would exit through the next face
        outside = ~np.any([inside(p, pts + eps * nrm, eps=0.0) for p in obj.parts], 0)
        inner = np.any([inside(p, pts - eps * nrm, eps=0.0) for p in obj.parts], 0)
        assert outside.mean() > 0.99 and inner.mean() > 0.99  # outward normals (points on edges may graze)
        # the simulated geoms agree: a ray cast back along -n from 2 mm outside hits the surface at 2 mm
        spec = mujoco.MjSpec()
        add_object(spec, 0, obj, 0.8)
        m = spec.compile()
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        origin = d.xpos[1]
        hits = [mujoco.mj_ray(m, d, origin + p + 0.002 * n, -n, None, 1, -1, np.zeros(1, np.int32), None)
                for p, n in zip(pts, nrm)]
        assert np.mean(np.abs(np.array(hits) - 0.002) < 5e-4) > 0.98, family


def test_library_split_and_masses():
    cfg = OmegaConf.load(REPO_CFG).objects
    objs = build_library(cfg)
    assert len(objs) == cfg.n_objects
    assert all(0 < o.mass <= cfg.max_mass for o in objs)
    train, held = split_indices(cfg, "train"), split_indices(cfg, "heldout")
    assert len(set(train) & set(held)) == 0 and len(train) + len(held) == cfg.n_objects
    again = build_library(cfg)
    assert all(np.array_equal(a.cloud, b.cloud) for a, b in zip(objs, again))  # seeded
