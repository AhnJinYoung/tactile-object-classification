"""Object gallery for the report page: renders every library object alone on the table (CPU MuJoCo, EGL) and writes
its facts, so the page can show each object with its size.

    MUJOCO_GL=egl python -m tacfind.report.gallery      # or: python scripts/report.py --objects

Output (git-ignored): runs/dev/objects/<name>_<view>.webp and objects.json. Per object: a three-quarter view
(perspective), a side and a top view (orthographic), and the Sharpa hand lying flat open beside it (orthographic, from
above). Scale reference in every image: a paper sheet with grid lines `GRID` m apart (bolder every `MAJOR` m) and a
vertical ruler of alternating `GRID` m segments, all geoms placed at exact positions in the scene.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from tacfind.envs import ycb
from tacfind.envs.bin_env import REPO, plain
from tacfind.envs.objects import build_library
from tacfind.envs.scene import FINGERS, HAND, add_object, add_table, hand_spec

OUT = REPO / "runs" / "dev" / "objects"
GRID, MAJOR, SHEET = 0.01, 0.05, 0.40   # grid spacing, bold-line spacing, sheet side, m
SIZE = (480, 360)                       # image width, height, px
VIEWS = ("threequarter", "side", "top", "hand")
HAND_GAP = 0.02                         # gap between the object and the hand, m
DROP, SETTLE, REST = 0.005, 5.0, 1e-3   # drop height (m), longest settling time (s), resting speed (|qvel|, SI)
QUALITY = 72                            # WebP quality


def pca_extents(cloud: np.ndarray, verts: np.ndarray) -> list[float]:
    """Extents (m) of `verts` (V, 3) along the principal axes of `cloud` (N, 3), sorted smallest first."""
    centred = cloud - cloud.mean(0)
    _, axes = np.linalg.eigh(centred.T @ centred)
    return sorted(float(v) for v in np.ptp(verts @ axes, 0))


def look_quat(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Camera quaternion (w x y z) looking from eye at target (MuJoCo cameras look along their -z, y up)."""
    back = (eye - target) / np.linalg.norm(eye - target)
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([right, np.cross(back, right), back]).ravel())
    return quat


VIS = dict(contype=0, conaffinity=0, type=mujoco.mjtGeom.mjGEOM_BOX, material="matte")  # visual-only boxes
# matte: without specular highlights the light straight above washes out the grid lines in the top view


def add_sheet(world: mujoco.MjsBody, lo: np.ndarray, hi: np.ndarray, axis: int, side: int) -> None:
    """Grid paper between corners lo, hi (3,) (world, m; lo[axis] == hi[axis] is the plane), lines `GRID` apart
    counted from lo (bold every `MAJOR`), drawn on the `side` (+1 / -1 along `axis`) facing the camera."""
    centre, half = (lo + hi) / 2, (hi - lo) / 2
    paper = centre.copy()
    paper[axis] += side * 1e-4
    world.add_geom(size=np.where(np.arange(3) == axis, 1e-4, half), pos=paper, rgba=[0.95, 0.94, 0.9, 1], **VIS)
    a, b = (i for i in range(3) if i != axis)
    for along, across in ((a, b), (b, a)):
        for k in range(round((hi[across] - lo[across]) / GRID) + 1):
            bold = round(k * GRID / MAJOR, 6).is_integer()
            size, pos = np.zeros(3), centre.copy()
            size[along], size[across], size[axis] = half[along], 9e-4 if bold else 4e-4, 5e-5
            pos[across], pos[axis] = lo[across] + k * GRID, lo[axis] + side * (2.5e-4 + 1e-5 * along)
            world.add_geom(size=size, pos=pos, rgba=[0.15, 0.15, 0.2, 1] if bold else [0.45, 0.47, 0.54, 1], **VIS)


def add_scale(spec: mujoco.MjSpec, centre: np.ndarray, top: float, ruler_at: np.ndarray, ruler_h: float) -> None:
    """Grid paper on the table around `centre` (x, y), a grid backdrop standing at its far (+y) edge (scale of the
    side view) and a vertical ruler of `GRID` segments at `ruler_at` (x, y); all visual only."""
    world, half = spec.worldbody, SHEET / 2
    spec.add_material(name="matte", specular=0.0, shininess=0.0, reflectance=0.0)
    add_sheet(world, np.array([*(centre - half), top]), np.array([*(centre + half), top]), 2, 1)
    back = centre[1] + half
    add_sheet(world, np.array([centre[0] - half, back, top]), np.array([centre[0] + half, back, top + half]), 1, -1)
    for k in range(round(ruler_h / GRID)):
        rgba = [0.1, 0.1, 0.1, 1] if k % 2 == 0 else [0.95, 0.95, 0.95, 1]
        world.add_geom(size=[2.5e-3, 2.5e-3, GRID / 2], pos=[*ruler_at, top + (k + 0.5) * GRID], rgba=rgba, **VIS)


def ruler(at: np.ndarray, extent: float) -> tuple[np.ndarray, float]:
    """Ruler position (x, y) behind and left of an object at `at` (x, y) with longest extent `extent`, and its height,
    m (whole `GRID` segments, at least 5, one more than the object needs)."""
    return at + [-0.6 * extent - 0.01, 0.6 * extent + 0.01], GRID * max(5, np.ceil(extent / GRID) + 1)


def scene(cfg, obj, at: np.ndarray, extent: float, hand_pose: tuple | None = None) -> mujoco.MjModel:
    """Table, scale sheet and ruler, the object as free body `obj0`, optionally the hand fixed at (pos, quat), and
    one camera `view` (posed per image by `shoot`)."""
    spec = mujoco.MjSpec()
    spec.option.timestep, spec.option.integrator = cfg.timestep, mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone, spec.option.impratio = mujoco.mjtCone.mjCONE_ELLIPTIC, 10.0
    spec.visual.global_.offwidth, spec.visual.global_.offheight = SIZE
    spec.visual.headlight.diffuse, spec.visual.headlight.ambient = [0.35] * 3, [0.25] * 3  # no white-out from above
    add_table(spec, cfg)
    add_scale(spec, at, cfg.table.top_z, *ruler(at, extent))
    add_object(spec, 0, obj, REPO / cfg.assets, textured=True)
    if hand_pose is not None:
        frame = spec.worldbody.add_frame(pos=hand_pose[0], quat=hand_pose[1])
        spec.attach(hand_spec(cfg, REPO / cfg.assets), frame=frame, prefix=HAND)
    spec.worldbody.add_camera(name="view")
    return spec.compile()


def settle(cfg, obj, verts: np.ndarray, at: np.ndarray, extent: float) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Drop the object upright (scan frame, z up) `DROP` above the table at `at` and simulate until its speed
    (norm of the free joint's linear m/s and angular rad/s velocity) falls below `REST`, at most `SETTLE` s."""
    m = scene(cfg, obj, at, extent)
    d = mujoco.MjData(m)
    d.qpos[:7] = [*at, cfg.table.top_z + DROP - verts[:, 2].min(), 1, 0, 0, 0]
    for _ in range(round(SETTLE / 0.25)):
        mujoco.mj_step(m, d, nstep=round(0.25 / cfg.timestep))
        if d.time >= 0.5 and np.linalg.norm(d.qvel[:6]) < REST:
            break
    return m, d


def world_verts(d: mujoco.MjData, verts: np.ndarray) -> np.ndarray:
    """Object mesh vertices (V, 3) in the world frame of the settled pose, m."""
    return verts @ d.xmat[1].reshape(3, 3).T + d.xpos[1]


def hand_flat(cfg) -> tuple[np.ndarray, np.ndarray, dict]:
    """The hand lying palm down, fingers along +y, joints at zero (flat open).

    Returns:
        rotation (3, 3) hand -> world, collision-mesh vertices (V, 3) in the rotated hand frame (m), and reference
        numbers in cm: thumb-index pad distance, length (along the fingers) and width of the flat hand.
    """
    spec = mujoco.MjSpec()
    spec.attach(hand_spec(cfg, REPO / cfg.assets), frame=spec.worldbody.add_frame(), prefix=HAND)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    rot = np.array([[0.0, -1, 0], [0, 0, 1], [-1, 0, 0]])  # hand x (palm normal) -> -z, hand z (fingers) -> +y
    verts = []
    for g in np.flatnonzero((m.geom_type == mujoco.mjtGeom.mjGEOM_MESH) & (m.geom_contype > 0)):
        v = m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]]:][:m.mesh_vertnum[m.geom_dataid[g]]]
        verts.append((v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]) @ rot.T)
    verts = np.vstack(verts)
    pad = {f: d.geom_xpos[m.geom(f"{HAND}right_{f}_elastomer").id] for f in FINGERS}
    ref = {"thumb_index_cm": 100 * float(np.linalg.norm(pad["thumb"] - pad["index"])),
           "length_cm": 100 * float(np.ptp(verts[:, 1])), "width_cm": 100 * float(np.ptp(verts[:, 0]))}
    return rot, verts, ref


def frame_views(lo: np.ndarray, hi: np.ndarray) -> dict:
    """{view: (eye, target, up, orthographic, fovy)} framing the box lo..hi (world, m); orthographic fovy in m."""
    c, size = (lo + hi) / 2, hi - lo
    aspect, pad = SIZE[0] / SIZE[1], 1.35
    az, el = np.radians(-50.0), np.radians(30.0)  # camera in front (-y) and to the right (+x), 30 deg above
    dist = 1.15 * float(np.linalg.norm(size)) / 2 / np.sin(np.radians(22.5))  # bounding sphere fills the fovy
    eye = c + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    z = np.array([0.0, 0, 1])
    return {
        "threequarter": (eye, c, z, 0, 45.0),
        "side": (c - [0, 1, 0], c, z, 1, pad * max(size[2], size[0] / aspect)),
        "top": (c + [0, 0, 1], c, np.array([0.0, 1, 0]), 1, pad * max(size[1], size[0] / aspect)),
    }


def shoot(m: mujoco.MjModel, d: mujoco.MjData, renderer: mujoco.Renderer, view: tuple) -> Image.Image:
    """Render the model's camera from one (eye, target, up, orthographic, fovy) view."""
    eye, target, up, ortho, fovy = view
    m.cam_pos[0], m.cam_quat[0] = eye, look_quat(eye, target, up)
    m.cam_projection[0], m.cam_fovy[0] = ortho, fovy  # projection 1: orthographic, fovy in m
    mujoco.mj_forward(m, d)
    renderer.update_scene(d, camera="view")
    return Image.fromarray(renderer.render())


def render_object(cfg, obj, split: str, hand: tuple, out: Path) -> dict:
    """Render the four images of one object into `out` and return its facts."""
    verts, _, _ = ycb.load_mesh(REPO / cfg.assets, obj.mesh)
    at = np.asarray(cfg.zone.center, float)
    extent = float(np.ptp(verts, 0).max())
    m, d = settle(cfg, obj, verts, at, extent)
    wv, top = world_verts(d, verts), cfg.table.top_z
    lo, hi = wv.min(0), wv.max(0)
    (rx, ry), rh = ruler(at, extent)
    box_lo, box_hi = np.minimum(lo, [rx, ry, top]), np.maximum(hi, [rx, ry, top + rh])  # object and ruler
    shots = {}
    with mujoco.Renderer(m, SIZE[1], SIZE[0]) as r:
        for name, view in frame_views(box_lo, box_hi).items():
            shots[name] = shoot(m, d, r, view)
    rot, hverts, _ = hand
    shift = np.array([lo[0] - HAND_GAP - hverts[:, 0].max(), (lo[1] + hi[1]) / 2 - hverts[:, 1].mean(),
                      top + 5e-4 - hverts[:, 2].min()])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.ravel())
    mh = scene(cfg, obj, at, extent, (shift, quat))
    dh = mujoco.MjData(mh)
    dh.qpos[:7] = d.qpos[:7]
    both = np.vstack([wv, hverts + shift])
    with mujoco.Renderer(mh, SIZE[1], SIZE[0]) as r:
        shots["hand"] = shoot(mh, dh, r, frame_views(both.min(0), both.max(0))["top"])
    for name, img in shots.items():
        img.save(out / f"{obj.family}_{name}.webp", quality=QUALITY, method=6)
    return {"name": obj.family, "split": split, "material": obj.material, "mass_g": 1000 * obj.mass,
            "mass_estimate": ycb.MASS_NOTE.get(obj.family) == "E",
            "dims_cm": [100 * v for v in pca_extents(obj.cloud[:, :3], verts)],
            "box_cm": sorted(100 * float(v) for v in np.ptp(verts, 0)),
            "rest_height_cm": 100 * float(hi[2] - top), "sink_mm": 1000 * max(0.0, float(top - lo[2])),
            "rest_speed": float(np.linalg.norm(d.qvel[:6]))}


def render_gallery(cfg, out: Path = OUT) -> Path:
    """Render every YCB library object and write `out/objects.json`.

    Args:
        cfg: `env` config node (objects.set must be ycb).
        out: cache directory.

    Returns:
        Path of objects.json: {scale: {grid_cm, major_cm, sheet_cm}, rest: {drop_mm, settle_s, speed}, hand_gap_cm,
        n_points (cloud size), hand: {thumb_index_cm, length_cm, width_cm},
        views, objects: [{name, split, material, mass_g, mass_estimate, dims_cm (PCA extents), box_cm (extents along
        the scan's own axes, the centred bounding box of the YCB file), both smallest first, rest_height_cm,
        sink_mm (how far the visual scan dips below the table top at rest: the collision hulls do not reach the
        scan's surface there), rest_speed (m/s, settling check)}]} in library order.
    """
    assert cfg.objects.set == "ycb", "the gallery renders the YCB scans"
    out.mkdir(parents=True, exist_ok=True)
    hand = hand_flat(cfg)
    split = {name: s for name, (*_, s) in ycb.OBJECTS.items()}
    objects = []
    for obj in build_library(cfg.objects, REPO / cfg.assets):
        objects.append(render_object(cfg, obj, split[obj.family], hand, out))
        print(f"  {obj.family:15s} {objects[-1]['dims_cm']} cm", flush=True)
    facts = {"scale": {"grid_cm": 100 * GRID, "major_cm": 100 * MAJOR, "sheet_cm": 100 * SHEET},
             "rest": {"drop_mm": 1000 * DROP, "settle_s": SETTLE, "speed": REST}, "hand_gap_cm": 100 * HAND_GAP,
             "n_points": cfg.objects.n_points,
             "hand": hand[2], "views": list(VIEWS), "objects": objects}
    (out / "objects.json").write_text(json.dumps(facts, indent=1))
    return out / "objects.json"


if __name__ == "__main__":
    env = plain(OmegaConf.load(REPO / "configs" / "env" / "bin.yaml"))
    print(f"wrote {render_gallery(env)}")
