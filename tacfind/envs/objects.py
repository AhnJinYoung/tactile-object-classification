"""Object library: YCB scans (`objects.set: ycb`, see `ycb.py`) or seeded procedural shapes (unions of primitives).

The library is the source of both the simulated objects (`scene.py` turns each entry into geoms) and the target point
clouds the networks match touches against (`clouds`, one `(N, 6)` cloud with outward normals per object).
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tacfind.envs import ycb

REPO = Path(__file__).resolve().parents[2]
_CACHE = {}

FAMILIES = ("box", "cylinder", "sphere", "capsule", "ellipsoid", "prism", "lblock", "tblock")


@dataclass(frozen=True)
class Part:
    """One convex primitive of an object, in the object frame.

    Attributes:
        kind: "box" | "cylinder" | "sphere" | "capsule" | "ellipsoid" | "prism".
        size: MuJoCo geom size (half-sizes, radius / half-length; for "prism" (side, half-length)), m.
        pos: (3,) centre in the object frame, m. Parts are axis-aligned with the object frame.
    """

    kind: str
    size: tuple
    pos: tuple = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class LibObject:
    """A library object: a union of parts with its mass and surface point cloud.

    Attributes:
        family: shape family (procedural) or YCB object name.
        parts: convex parts of a procedural object (overlapping parts form a union); () for a YCB object.
        mass: total mass, kg.
        cloud: (N, 6) float32 surface points (m) and outward unit normals, object (= body) frame.
        mesh: YCB object name (its mesh directory), "" for procedural objects.
        material: material class (friction source), "procedural" for procedural objects.
        friction: sliding friction of the object's geoms (object vs table / other objects).
    """

    family: str
    parts: tuple
    mass: float
    cloud: np.ndarray
    mesh: str = ""
    material: str = "procedural"
    friction: float = 0.8


def prism_vertices(side: float) -> np.ndarray:
    """Vertices (3, 2) of the equilateral triangle cross-section of a prism, centred on its centroid, m."""
    ang = np.pi / 2 + np.array([0.0, 2.0, 4.0]) * np.pi / 3
    return side / np.sqrt(3.0) * np.stack([np.cos(ang), np.sin(ang)], 1)


def make_parts(family: str, dims: np.ndarray, thick: float) -> tuple:
    """Parts of one object from three full extents `dims` (m) and a bar-thickness ratio for L- and T-blocks."""
    a, b, c = dims / 2  # half extents
    if family == "box":
        return (Part("box", (a, b, c)),)
    if family == "cylinder":
        return (Part("cylinder", (a, c)),)
    if family == "sphere":
        return (Part("sphere", (a,)),)
    if family == "capsule":
        return (Part("capsule", (a, max(c - a, 0.25 * a))),)
    if family == "ellipsoid":
        return (Part("ellipsoid", (a, b, c)),)
    if family == "prism":
        return (Part("prism", (dims[0], c)),)
    t = thick * min(a, b)  # half thickness of a bar; bars overlap so the union has no internal faces on its surface
    if family == "lblock":  # horizontal bar along x plus vertical bar at its -x end, overlapping in the corner
        return (Part("box", (a, t, c), (0.0, -b + t, 0.0)), Part("box", (t, b, c), (-a + t, 0.0, 0.0)))
    if family == "tblock":  # top bar along x plus a stem down the middle
        return (Part("box", (a, t, c), (0.0, b - t, 0.0)), Part("box", (t, b, c), (0.0, 0.0, 0.0)))
    raise ValueError(f"unknown family {family}")


def volume(p: Part) -> float:
    """Volume of one part, m^3."""
    s = p.size
    return {
        "box": lambda: 8 * s[0] * s[1] * s[2],
        "cylinder": lambda: 2 * np.pi * s[0] ** 2 * s[1],
        "sphere": lambda: 4 / 3 * np.pi * s[0] ** 3,
        "capsule": lambda: np.pi * s[0] ** 2 * (2 * s[1] + 4 / 3 * s[0]),
        "ellipsoid": lambda: 4 / 3 * np.pi * s[0] * s[1] * s[2],
        "prism": lambda: np.sqrt(3) / 4 * s[0] ** 2 * 2 * s[1],
    }[p.kind]()


def area(p: Part) -> float:
    """Surface area of one part, m^2 (Knud Thomsen's approximation for ellipsoids, error < 1.1 %)."""
    s, k = p.size, 1.6075
    return {
        "box": lambda: 8 * (s[0] * s[1] + s[1] * s[2] + s[0] * s[2]),
        "cylinder": lambda: 4 * np.pi * s[0] * s[1] + 2 * np.pi * s[0] ** 2,
        "sphere": lambda: 4 * np.pi * s[0] ** 2,
        "capsule": lambda: 4 * np.pi * s[0] * s[1] + 4 * np.pi * s[0] ** 2,
        "ellipsoid": lambda: 4 * np.pi * (((s[0] * s[1]) ** k + (s[1] * s[2]) ** k + (s[0] * s[2]) ** k) / 3)
        ** (1 / k),
        "prism": lambda: np.sqrt(3) / 2 * s[0] ** 2 + 6 * s[0] * s[1],
    }[p.kind]()


def inside(p: Part, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Whether points `x` (M, 3) lie strictly inside part `p` (deeper than `eps`, m). Returns (M,) bool."""
    q, s = x - np.asarray(p.pos), p.size
    if p.kind == "box":
        return np.all(np.abs(q) < np.asarray(s) - eps, 1)
    if p.kind == "sphere":
        return np.linalg.norm(q, axis=1) < s[0] - eps
    if p.kind == "cylinder":
        return (np.linalg.norm(q[:, :2], axis=1) < s[0] - eps) & (np.abs(q[:, 2]) < s[1] - eps)
    if p.kind == "capsule":
        axis = q - np.clip(q[:, 2], -s[1], s[1])[:, None] * np.array([0, 0, 1.0])
        return np.linalg.norm(axis, axis=1) < s[0] - eps
    if p.kind == "ellipsoid":  # scaled radius, conservative eps
        return np.linalg.norm(q / np.asarray(s), axis=1) < 1 - eps / max(s)
    v = prism_vertices(s[0])
    edges = [(v[i], v[(i + 1) % 3]) for i in range(3)]
    left = [(e1 - e0)[0] * (q[:, 1] - e0[1]) - (e1 - e0)[1] * (q[:, 0] - e0[0]) for e0, e1 in edges]  # 2-D cross
    in_tri = np.all([c > eps * np.linalg.norm(e1 - e0) for c, (e0, e1) in zip(left, edges)], 0)
    return in_tri & (np.abs(q[:, 2]) < s[1] - eps)


def sample_surface(p: Part, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform (area-weighted) samples on the surface of one part. Returns (n, 6): points (m) and outward normals."""
    s = p.size
    if p.kind == "box":
        h = np.asarray(s)
        area = np.array([h[1] * h[2], h[0] * h[2], h[0] * h[1]]).repeat(2)
        face = rng.choice(6, n, p=area / area.sum())
        axis, sign = face // 2, np.where(face % 2, 1.0, -1.0)
        x = rng.uniform(-1, 1, (n, 3)) * h
        x[np.arange(n), axis] = sign * h[axis]
        nrm = np.zeros((n, 3))
        nrm[np.arange(n), axis] = sign
    elif p.kind in ("sphere", "ellipsoid"):
        h = np.full(3, s[0]) if p.kind == "sphere" else np.asarray(s)
        u = rng.normal(size=(8 * n, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        dens = np.linalg.norm(u / h, axis=1)  # area element of the ellipsoid relative to the unit sphere
        u = u[rng.uniform(0, dens.max(), len(u)) < dens][:n]
        x, nrm = u * h, u / h
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    elif p.kind in ("cylinder", "capsule"):
        r, hl = s
        cap = np.pi * r**2 if p.kind == "cylinder" else 2 * np.pi * r**2  # disc or hemisphere area
        side = rng.uniform(0, 4 * np.pi * r * hl + 2 * cap, n) < 4 * np.pi * r * hl
        phi = rng.uniform(0, 2 * np.pi, n)
        x = np.stack([r * np.cos(phi), r * np.sin(phi), rng.uniform(-hl, hl, n)], 1)
        nrm = np.stack([np.cos(phi), np.sin(phi), np.zeros(n)], 1)
        sign = np.where(rng.uniform(size=n) < 0.5, 1.0, -1.0)
        if p.kind == "cylinder":
            rad = r * np.sqrt(rng.uniform(size=n))
            cap_x = np.stack([rad * np.cos(phi), rad * np.sin(phi), sign * hl], 1)
            cap_n = np.stack([np.zeros(n), np.zeros(n), sign], 1)
        else:
            u = rng.normal(size=(n, 3))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            u[:, 2] = np.abs(u[:, 2]) * sign
            cap_x, cap_n = u * r + np.outer(sign * hl, [0, 0, 1]), u
        x, nrm = np.where(side[:, None], x, cap_x), np.where(side[:, None], nrm, cap_n)
    else:  # prism: two triangles and three rectangles
        v, hl = prism_vertices(s[0]), s[1]
        tri, rect = np.sqrt(3) / 4 * s[0] ** 2, s[0] * 2 * hl
        face = rng.choice(5, n, p=np.array([tri, tri, rect, rect, rect]) / (2 * tri + 3 * rect))
        a, b = rng.uniform(size=(2, n))
        flip = a + b > 1
        a, b = np.where(flip, 1 - a, a), np.where(flip, 1 - b, b)
        xy_tri = v[0] + a[:, None] * (v[1] - v[0]) + b[:, None] * (v[2] - v[0])
        e = np.clip(face - 2, 0, 2)
        e0, e1 = v[e], v[(e + 1) % 3]
        xy_rect = e0 + a[:, None] * (e1 - e0)
        out = (e1 - e0) @ np.array([[0, -1], [1, 0]])  # rotate edge by -90 deg: outward for a CCW triangle
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        is_tri = face < 2
        z = np.where(is_tri, np.where(face == 0, -hl, hl), rng.uniform(-hl, hl, n))
        x = np.column_stack([np.where(is_tri[:, None], xy_tri, xy_rect), z])
        nrm = np.column_stack([np.where(is_tri[:, None], 0.0, out), np.where(is_tri, np.where(face == 0, -1, 1), 0)])
    return np.hstack([x + np.asarray(p.pos), nrm])


def surface_cloud(parts: tuple, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 6) float32 points + outward normals on the outer surface of a union of parts (none inside another part)."""
    areas = np.array([area(p) for p in parts])
    chunks = []
    for i, p in enumerate(parts):  # area-proportional oversampling, then drop what is buried in another part
        x = sample_surface(p, int(8 * n * areas[i] / areas.sum()) + 1, rng)
        for j, other in enumerate(parts):
            x = x if j == i else x[~inside(other, x[:, :3])]
        chunks.append(x)
    pts = np.vstack(chunks)
    return pts[rng.choice(len(pts), n, replace=False)].astype(np.float32)


def build_library(cfg, root: Path = REPO / "third_party") -> list[LibObject]:
    """Build the object library (cached per configuration).

    Args:
        cfg: `env.objects` config node (set, n_points, seed; procedural: n_objects, families, extent, thickness,
            density, max_mass, friction).
        root: fetched assets (YCB meshes).

    Returns:
        YCB: the objects of `ycb.OBJECTS` in table order. Procedural: n_objects objects, families cycled in config
        order, sizes drawn uniformly from the config ranges.
    """
    key = (getattr(cfg, "set", "procedural"), cfg.seed, cfg.n_points, getattr(cfg, "n_objects", 0), str(root))
    if key not in _CACHE:
        _CACHE[key] = _build_ycb(cfg, root) if key[0] == "ycb" else _build_procedural(cfg)
    return _CACHE[key]


def _build_ycb(cfg, root: Path) -> list[LibObject]:
    """YCB library: measured masses, material friction, clouds sampled from the visual scans."""
    rng, lib = np.random.default_rng(cfg.seed), []
    for name, (mass, material, _) in ycb.OBJECTS.items():
        verts, faces, _ = ycb.load_mesh(root, name)
        cloud = ycb.sample_mesh(verts, faces, cfg.n_points, rng)
        lib.append(LibObject(name, (), mass, cloud, mesh=name, material=material, friction=ycb.FRICTION[material]))
    return lib


def _build_procedural(cfg) -> list[LibObject]:
    """Seeded procedural library (unions of primitives)."""
    rng = np.random.default_rng(cfg.seed)
    lib = []
    for i in range(cfg.n_objects):
        family = cfg.families[i % len(cfg.families)]
        dims = rng.uniform(*cfg.extent, 3)
        parts = make_parts(family, dims, rng.uniform(*cfg.thickness))
        mass = min(cfg.density * sum(volume(p) for p in parts), cfg.max_mass)  # overlaps make this a slight overcount
        lib.append(LibObject(family, parts, float(mass), surface_cloud(parts, cfg.n_points, rng),
                             friction=cfg.friction))
    return lib


def split_indices(cfg, split: str) -> np.ndarray:
    """Library indices of the "train" or "heldout" split. YCB: the split column of `ycb.OBJECTS`; procedural: a
    seeded permutation of which `heldout_frac` is held out."""
    if getattr(cfg, "set", "procedural") == "ycb":
        return np.array([i for i, (*_, s) in enumerate(ycb.OBJECTS.values()) if s == split])
    perm = np.random.default_rng(cfg.seed + 1).permutation(cfg.n_objects)
    n_held = int(round(cfg.heldout_frac * cfg.n_objects))
    return np.sort(perm[:n_held] if split == "heldout" else perm[n_held:])
