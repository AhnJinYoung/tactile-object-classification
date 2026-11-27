"""Build the MuJoCo scene: UR10e + Sharpa Wave hand on a table with an open-top bin, plus every library object.

One model serves all worlds: each world activates `n_slots` objects in the bin; the others wait at parking spots far
below the table, spaced apart, their weight cancelled by an applied force (set by `BinEnv`), so they never touch
anything and add no constraints.
"""

from pathlib import Path

import mujoco
import numpy as np

from tacfind.envs.objects import LibObject, prism_vertices, volume

ARM_XML = "mujoco_menagerie/universal_robots_ur10e/ur10e.xml"
HAND_XML = "sharpa/wave_01/right_sharpa_wave/right_sharpa_wave_with_flange.xml"
HAND = "hand/"  # name prefix of every hand element
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
# Collision bitmasks (contact iff contype_a & conaffinity_b or contype_b & conaffinity_a): robot-robot contacts are
# off (saves the costly finger mesh-mesh tests), everything else collides.
ROBOT, STATIC, OBJECT = 1, 2, 4
GEOM_TYPE = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    "prism": mujoco.mjtGeom.mjGEOM_MESH,
}
RGBA = ((0.85, 0.35, 0.25, 1), (0.25, 0.55, 0.85, 1), (0.95, 0.75, 0.2, 1), (0.6, 0.4, 0.8, 1), (0.9, 0.5, 0.7, 1))


def park_position(j: int) -> np.ndarray:
    """Parking spot (3,) of object body j, base frame, m: a grid 3 m away from the robot, 0.3 m apart."""
    return np.array([-3.0 + 0.3 * (j % 10), 3.0 + 0.3 * (j // 10), -1.0])


def robot_spec(cfg, root: Path) -> mujoco.MjSpec:
    """UR10e with the Sharpa Wave on its flange, ideal gravity compensation, flange F/T sensor and a palm site."""
    arm = mujoco.MjSpec.from_file(str(root / ARM_XML))
    hand = mujoco.MjSpec.from_file(str(root / HAND_XML))
    hand_root = hand.worldbody.first_body()
    hand_root.add_site(name="flange", group=4)
    hand_root.add_site(name="tcp", pos=[0.03, 0.0, 0.16], group=4)  # in front of the palm, between the fingertips
    site = arm.site("attachment_site")
    site.quat = _quat_mul(np.asarray(site.quat, float), np.asarray(cfg.hand_mount_quat, float))
    arm.attach(hand, site=site, prefix=HAND)
    for body in arm.bodies:
        body.gravcomp = 1.0  # ideal gravity compensation, as the UR and Sharpa controllers do on the real robot
    for geom in arm.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype, geom.conaffinity = ROBOT, STATIC | OBJECT
    for kind in ("force", "torque"):
        arm.add_sensor(name=f"flange_{kind}", type=getattr(mujoco.mjtSensor, f"mjSENS_{kind.upper()}"),
                       objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"{HAND}flange")
    for key in arm.keys:  # the menagerie keyframe does not cover the objects added later
        arm.delete(key)
    return arm


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two (w, x, y, z) quaternions."""
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def add_bin(spec: mujoco.MjSpec, cfg) -> None:
    """Table top and an open-top bin (floor + four walls) as static world geoms; walls are see-through in renders."""
    b, world = cfg.bin, spec.worldbody
    cx, cy = b.center
    (w, d), t, h = (np.asarray(b.inner) / 2), b.wall_thickness, b.wall_height
    kw = dict(contype=STATIC, conaffinity=ROBOT | OBJECT, type=mujoco.mjtGeom.mjGEOM_BOX)
    world.add_geom(name="table", size=[*np.asarray(cfg.table_size) / 2, 0.02], pos=[cx, cy, b.floor_z - t - 0.02],
                   rgba=[0.55, 0.45, 0.35, 1], **kw)
    world.add_geom(name="bin_floor", size=[w + t, d + t, t / 2], pos=[cx, cy, b.floor_z - t / 2],
                   rgba=[0.3, 0.3, 0.32, 1], **kw)
    wall = [0.5, 0.6, 0.7, b.wall_alpha]
    for i, (sx, sy, px, py) in enumerate([(t / 2, d + t, w + t / 2, 0), (w + t, t / 2, 0, d + t / 2)] * 2):
        sign = 1 if i < 2 else -1
        world.add_geom(name=f"bin_wall{i}", size=[sx, sy, h / 2], pos=[cx + sign * px, cy + sign * py,
                       b.floor_z + h / 2], rgba=wall, **kw)
    world.add_light(pos=[cx, cy, 2.0], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6], castshadow=False)


def add_object(spec: mujoco.MjSpec, j: int, obj: LibObject, friction: float) -> None:
    """Library object as free body `obj{j}` at its parking spot; geom densities are scaled so its mass is `obj.mass`."""
    body = spec.worldbody.add_body(name=f"obj{j}", pos=park_position(j))
    body.add_freejoint(name=f"obj{j}")
    density = obj.mass / sum(volume(p) for p in obj.parts)
    for k, part in enumerate(obj.parts):
        # condim 6 (torsional + rolling friction): round objects come to rest instead of rocking for seconds
        kw = dict(type=GEOM_TYPE[part.kind], pos=part.pos, density=density, friction=[friction, 0.01, 0.002], condim=6,
                  contype=OBJECT, conaffinity=ROBOT | STATIC | OBJECT, rgba=RGBA[j % len(RGBA)])
        if part.kind == "prism":
            v = prism_vertices(part.size[0])
            verts = np.vstack([np.column_stack([v, np.full(3, z)]) for z in (-part.size[1], part.size[1])])
            spec.add_mesh(name=f"prism{j}_{k}", uservert=verts.ravel().tolist())
            body.add_geom(name=f"obj{j}_{k}", meshname=f"prism{j}_{k}", **kw)
        else:
            body.add_geom(name=f"obj{j}_{k}", size=list(part.size) + [0] * (3 - len(part.size)), **kw)


def build_scene(cfg, root: Path, objects: list[LibObject]) -> mujoco.MjModel:
    """Compile the scene model.

    Args:
        cfg: `env` config node.
        root: directory holding the fetched assets (`third_party/`).
        objects: the library objects of the split; object j becomes free body `obj{j}`.

    Returns:
        The compiled model: robot joints first (6 arm, 22 hand), then one free joint per object.
    """
    spec = robot_spec(cfg, root)
    spec.option.timestep = cfg.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC  # elliptic cones + high impratio: grasps slip less
    spec.option.impratio = 10.0
    spec.visual.global_.offwidth, spec.visual.global_.offheight = cfg.render.width, cfg.render.height
    add_bin(spec, cfg)
    for j, obj in enumerate(objects):
        add_object(spec, j, obj, cfg.objects.friction)
    return spec.compile()
