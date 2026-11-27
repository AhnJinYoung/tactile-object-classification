"""YCB object library: which objects, their measured masses, material classes and friction, and how to load them.

Meshes come from github.com/elpis-lab/YCB_Dataset (repo MIT; YCB models CC BY 4.0, ycbbenchmarks.com), fetched by
`scripts/fetch_assets.py`: Google 16k textured scans (`textured.obj` + `texture_map.png`, origin at the centre of the
oriented bounding box) and CoACD convex decompositions (<= 5 hulls of 32 faces each) for collision.

Sources of the numbers below:
  [M] mass: YCB object list (ycbbenchmarks.com/wp-content/uploads/2015/09/object-list-Sheet1.pdf), as copied into
      `scripts/ycb_mass.json` of the mesh repository. Lego masses are given per bag entry and look high for a single
      Duplo brick: marked estimate (E) in `MASS_NOTE`.
  [F] friction, object vs table (wood laminate): Engineering ToolBox, "Friction and friction coefficients for some
      common materials" (engineeringtoolbox.com/friction-coefficients-d_778.html), clean and dry. Ranges there:
      wood/wood 0.25-0.5, wood/clean metal 0.2-0.6, polystyrene/steel 0.3-0.35, nylon/steel 0.4, leather/wood 0.3-0.4,
      rubber/cardboard 0.5-0.8, rubber/dry concrete 0.6-0.85, paper/cast iron 0.2. No table gives plastic on wood or
      rubber on wood directly, so every class value is an estimate (E) inside or next to those ranges.
Selection rule (stated limits): longest extent <= 0.11 m (the Sharpa pinch closes on ~2-9 cm), mass <= 0.2 kg,
shortest extent >= 0.02 m, rigid single objects only. Excluded from the candidates fetched for the check: banana,
scissors, large_marker, large_clamp, pudding_box (too long), softball, mini_soccer_ball (too large), padlock,
potted_meat_can (too heavy), dice (too small), sponge, foam_brick (soft), toy airplane / wood block sets (mass is for
the whole set).
"""

from pathlib import Path

import mujoco
import numpy as np

MESH_DIR = "ycb/ycb"  # under third_party/

FRICTION = {"plastic": 0.35, "rubber": 0.7, "leather": 0.35, "metal": 0.4, "cardboard": 0.35}  # [F], all (E)

# name: (mass kg [M], material class, split)
OBJECTS = {
    "apple": (0.068, "plastic", "train"),      # YCB fruits are hollow plastic replicas
    "lemon": (0.029, "plastic", "train"),
    "orange": (0.047, "plastic", "train"),
    "peach": (0.033, "plastic", "train"),
    "pear": (0.049, "plastic", "train"),
    "plum": (0.025, "plastic", "heldout"),
    "strawberry": (0.018, "plastic", "train"),
    "golf_ball": (0.046, "plastic", "train"),
    "racquetball": (0.041, "rubber", "heldout"),
    "tennis_ball": (0.058, "rubber", "train"),   # felt cover: rubber class (E)
    "baseball": (0.138, "leather", "train"),
    "rubiks_cube": (0.094, "plastic", "heldout"),
    "gelatin_box": (0.097, "cardboard", "train"),
    "tuna_fish_can": (0.171, "metal", "train"),
    "mug": (0.118, "plastic", "train"),
    "medium_clamp": (0.059, "plastic", "train"),
    "a_lego_duplo": (0.0372, "plastic", "train"),
    "d_lego_duplo": (0.0372, "plastic", "heldout"),
    "e_lego_duplo": (0.0372, "plastic", "train"),
    "a_cups": (0.013, "plastic", "train"),
    "e_cups": (0.021, "plastic", "heldout"),
}
MASS_NOTE = {"a_lego_duplo": "E", "d_lego_duplo": "E", "e_lego_duplo": "E"}


def load_mesh(root: Path, name: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Visual mesh of one object in its file (= body) frame.

    Returns:
        vertices (V, 3) m, faces (F, 3) int, enclosed volume m^3 (MuJoCo's mesh volume; scans are closed surfaces).
    """
    spec = mujoco.MjSpec()
    spec.compiler.meshdir = str(root / MESH_DIR / name)
    spec.add_mesh(name="v", file="textured.obj")
    spec.worldbody.add_body().add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="v", density=1.0)
    m = spec.compile()
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, m.geom_quat[0])  # MuJoCo centres meshes at their centroid; undo it
    verts = m.mesh_vert[:m.mesh_vertnum[0]] @ rot.reshape(3, 3).T + m.geom_pos[0]
    return verts, m.mesh_face[:m.mesh_facenum[0]].copy(), float(m.body_mass[1])


def sample_mesh(verts: np.ndarray, faces: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Area-weighted surface samples (n, 6) float32: points (m) and outward unit face normals (scan winding)."""
    tri = verts[faces]  # (F, 3, 3)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    f = rng.choice(len(faces), n, p=area / area.sum())
    a, b = rng.uniform(size=(2, n))
    flip = a + b > 1
    a, b = np.where(flip, 1 - a, a), np.where(flip, 1 - b, b)
    pts = tri[f, 0] + a[:, None] * (tri[f, 1] - tri[f, 0]) + b[:, None] * (tri[f, 2] - tri[f, 0])
    return np.hstack([pts, cross[f] / area[f, None]]).astype(np.float32)
