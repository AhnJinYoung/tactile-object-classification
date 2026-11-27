"""Object libraries: surface clouds lie on the simulated surface with unit outward normals (procedural and YCB)."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
from omegaconf import OmegaConf

from tacfind.envs import ycb
from tacfind.envs.objects import FAMILIES, build_library, inside, split_indices
from tacfind.envs.scene import add_object

REPO = Path(__file__).resolve().parents[1]
CFG = REPO / "configs" / "env" / "bin.yaml"


def ray_hits(obj, root: Path, group: int) -> np.ndarray:
    """Distances of rays cast from 2 mm outside each cloud point back along -normal, against geom `group` only."""
    spec = mujoco.MjSpec()
    add_object(spec, 0, obj, root, textured=False)
    m = spec.compile()
    d = mujoco.MjData(m)
    d.qpos[:3] = 0.0
    mujoco.mj_forward(m, d)
    groups = np.zeros(6, np.uint8)
    groups[group] = 1
    pts, nrm = obj.cloud[:, :3].astype(np.float64), obj.cloud[:, 3:].astype(np.float64)
    return np.array([mujoco.mj_ray(m, d, p + 0.002 * n, -n, groups, 1, -1, np.zeros(1, np.int32), None)
                     for p, n in zip(pts, nrm)])


@pytest.fixture(scope="module")
def procedural():
    cfg = OmegaConf.load(CFG).objects
    cfg.set, cfg.n_objects = "procedural", 2 * len(FAMILIES)  # every family twice
    return cfg, build_library(cfg)


@pytest.mark.parametrize("family", FAMILIES)
def test_procedural_cloud_on_surface(procedural, family):
    cfg, objs = procedural
    for obj in [o for o in objs if o.family == family]:
        cloud = obj.cloud.astype(np.float64)
        assert cloud.shape == (cfg.n_points, 6) and obj.cloud.dtype == np.float32
        pts, nrm = cloud[:, :3], cloud[:, 3:]
        assert np.allclose(np.linalg.norm(nrm, axis=1), 1.0, atol=1e-5)
        eps = 1e-4  # small: a step inward near a sharp (60 deg prism) edge would exit through the next face
        outside = ~np.any([inside(p, pts + eps * nrm, eps=0.0) for p in obj.parts], 0)
        inner = np.any([inside(p, pts - eps * nrm, eps=0.0) for p in obj.parts], 0)
        assert outside.mean() > 0.99 and inner.mean() > 0.99  # outward normals (points on edges may graze)
        hits = ray_hits(obj, REPO / "third_party", group=0)  # the simulated geoms agree
        assert np.mean(np.abs(hits - 0.002) < 5e-4) > 0.98, family


def test_procedural_split_and_masses(procedural):
    cfg = OmegaConf.load(CFG).objects
    cfg.set = "procedural"
    objs = build_library(cfg)
    assert len(objs) == cfg.n_objects and all(0 < o.mass <= cfg.max_mass for o in objs)
    train, held = split_indices(cfg, "train"), split_indices(cfg, "heldout")
    assert len(set(train) & set(held)) == 0 and len(train) + len(held) == cfg.n_objects


def test_ycb_clouds_masses_and_split():
    root = REPO / "third_party"
    if not (root / ycb.MESH_DIR).is_dir():
        pytest.skip("YCB assets missing: run scripts/fetch_assets.py")
    cfg = OmegaConf.load(CFG).objects
    objs = build_library(cfg, root)
    assert [o.family for o in objs] == list(ycb.OBJECTS) and all(0 < o.mass <= 0.2 for o in objs)
    extent = np.array([np.ptp(o.cloud[:, :3], 0) for o in objs])
    assert (extent.max(1) <= 0.11).all() and (extent.min(1) >= 0.02).all()  # the stated selection limits
    train, held = split_indices(cfg, "train"), split_indices(cfg, "heldout")
    assert len(set(train) & set(held)) == 0 and len(train) + len(held) == len(objs)
    for obj in objs[::4]:  # points on the visual scan (group 0), normals unit and outward
        assert np.allclose(np.linalg.norm(obj.cloud[:, 3:], axis=1), 1.0, atol=1e-4)
        hits = ray_hits(obj, root, group=0)
        assert np.mean(np.abs(hits - 0.002) < 5e-4) > 0.95, obj.family
