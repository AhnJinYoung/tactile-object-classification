"""Build the MuJoCo scene: UR10e + Sharpa Wave hand at the long edge of a table, plus every library object.

One model serves all worlds: each world activates 3-8 objects on the table; the others wait at parking spots far
from the robot, spaced apart, their weight cancelled by an applied force (set by `BinEnv`), so they never touch
anything and add no constraints.
"""

import re
from pathlib import Path

import mujoco
import numpy as np

from tacfind.envs import ycb
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
# Real Sharpa Wave joint torque limits, N m, by joint-name pattern (first match wins). With the official MJCF gains a
# pressed fingertip gives way at a few N, while the real hand presses up to ~20 N (`finger_gains: torque_limit`).
TORQUE_LIMIT = (("thumb_CMC", 5.0), ("thumb_MCP_FE", 2.8), ("thumb_MCP_AA", 2.6), ("pinky_CMC", 0.5), ("MCP_FE", 2.15),
                ("MCP_AA", 2.0), ("_(PIP|IP)$", 1.06), ("_DIP$", 0.21))
RGBA = ((0.85, 0.35, 0.25, 1), (0.25, 0.55, 0.85, 1), (0.95, 0.75, 0.2, 1), (0.6, 0.4, 0.8, 1), (0.9, 0.5, 0.7, 1))


def park_position(j: int) -> np.ndarray:
    """Parking spot (3,) of object body j, base frame, m: a grid 2 m up and 3 m behind the robot, 0.3 m apart."""
    return np.array([3.0 + 0.3 * (j % 10), 0.3 * (j // 10), 2.0])


def hand_spec(cfg, root: Path) -> mujoco.MjSpec:
    """Sharpa Wave right hand (flange variant) with `flange` and `tcp` sites and the configured finger gains.

    `finger_gains: mjcf` keeps the official position gains; `torque_limit` sets kp = limit / 0.01 rad and
    forcerange = +-limit per joint (TORQUE_LIMIT); the MJCF dampratio stays, so damping scales with sqrt(kp).
    """
    hand = mujoco.MjSpec.from_file(str(root / HAND_XML))
    hand_root = hand.worldbody.first_body()
    hand_root.add_site(name="flange", group=4)
    hand_root.add_site(name="tcp", pos=[0.03, 0.0, 0.16], group=4)  # in front of the palm, between the fingertips
    if cfg.finger_gains == "torque_limit":
        for act in hand.actuators:
            limit = next(v for pattern, v in TORQUE_LIMIT if re.search(pattern, act.target))
            act.gainprm[0], act.biasprm[1] = limit / 0.01, -limit / 0.01  # biasprm[2] keeps the dampratio
            act.forcelimited, act.forcerange = mujoco.mjtLimited.mjLIMITED_TRUE, [-limit, limit]
    else:
        assert cfg.finger_gains == "mjcf", cfg.finger_gains
    return hand


def robot_spec(cfg, root: Path) -> mujoco.MjSpec:
    """UR10e with the Sharpa Wave on its flange, ideal gravity compensation and a flange F/T sensor."""
    arm = mujoco.MjSpec.from_file(str(root / ARM_XML))
    hand = hand_spec(cfg, root)
    site = arm.site("attachment_site")
    site.quat = _quat_mul(np.asarray(site.quat, float), np.asarray(cfg.hand_mount_quat, float))
    arm.attach(hand, site=site, prefix=HAND)
    for body in arm.bodies:
        body.gravcomp = 1.0  # ideal gravity compensation, as the UR and Sharpa controllers do on the real robot
    for geom in arm.geoms:
        if geom.contype or geom.conaffinity:  # priority 2: finger contacts keep the Sharpa pad parameters
            geom.contype, geom.conaffinity, geom.priority = ROBOT, STATIC | OBJECT, 2
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


def add_table(spec: mujoco.MjSpec, cfg) -> None:
    """Table (top at `table.top_z`, a long edge at the robot base) and a floor plane for objects that fall off."""
    t, world = cfg.table, spec.worldbody
    sx, sy = np.asarray(t.size) / 2
    kw = dict(contype=STATIC, conaffinity=OBJECT, friction=[t.friction, 0.005, 0.0001])  # objects win, see add_object
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, 0.02],
                   pos=[t.edge_x - sx, 0.0, t.top_z - 0.02], rgba=[0.62, 0.5, 0.38, 1], **kw)
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], pos=[0, 0, cfg.floor_z],
                   rgba=[0.25, 0.25, 0.27, 1], **kw)
    cx, cy = cfg.zone.center
    world.add_light(pos=[cx, cy, 2.0], dir=[0, 0, -1], diffuse=[0.5, 0.5, 0.5], castshadow=False)
    world.add_light(pos=[cx - 1.0, cy + 1.0, 1.5], dir=[0.6, -0.6, -0.5], diffuse=[0.35, 0.35, 0.35], castshadow=False)


def add_object(spec: mujoco.MjSpec, j: int, obj: LibObject, root: Path, textured: bool) -> None:
    """Library object as free body `obj{j}` at its parking spot, with its measured mass.

    Contact parameters: object geoms have priority 1 over the table and floor (0), so the object's material friction
    decides object-table and object-object contacts; the hand's geoms have priority 2, so finger contacts use the
    Sharpa MJCF pad parameters. condim 6 (torsional + rolling friction) lets round objects come to rest instead of
    rocking and spinning for seconds. Inertia and centre of mass: a uniform-density solid of the visual scan (YCB) or
    of the primitives (procedural), scaled to the object's mass.
    """
    body = spec.worldbody.add_body(name=f"obj{j}", pos=park_position(j))
    body.add_freejoint(name=f"obj{j}")
    col = dict(contype=OBJECT, conaffinity=ROBOT | STATIC | OBJECT, condim=6, priority=1,
               friction=[obj.friction, 0.01, 0.002])
    if obj.mesh:
        d = root / ycb.MESH_DIR / obj.mesh
        spec.add_mesh(name=f"obj{j}_visual", file=str(d / "textured.obj"))
        vis = body.add_geom(name=f"obj{j}_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"obj{j}_visual",
                            mass=obj.mass, contype=0, conaffinity=0, group=0, rgba=[0.8, 0.8, 0.8, 1])
        if textured:
            spec.add_texture(name=f"obj{j}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(d / "texture_map.png"))
            mat = spec.add_material(name=f"obj{j}_mat")
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"obj{j}_tex"
            vis.material, vis.rgba = f"obj{j}_mat", [1, 1, 1, 1]
        for k, hull in enumerate(sorted(d.glob("textured_coacd_*.stl"))):
            spec.add_mesh(name=f"obj{j}_hull{k}", file=str(hull))
            body.add_geom(name=f"obj{j}_{k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"obj{j}_hull{k}",
                          density=0.0, group=3, rgba=RGBA[j % len(RGBA)], **col)
        return
    density = obj.mass / sum(volume(p) for p in obj.parts)
    for k, part in enumerate(obj.parts):
        kw = dict(type=GEOM_TYPE[part.kind], pos=part.pos, density=density, rgba=RGBA[j % len(RGBA)], **col)
        if part.kind == "prism":
            v = prism_vertices(part.size[0])
            verts = np.vstack([np.column_stack([v, np.full(3, z)]) for z in (-part.size[1], part.size[1])])
            spec.add_mesh(name=f"prism{j}_{k}", uservert=verts.ravel().tolist())
            body.add_geom(name=f"obj{j}_{k}", meshname=f"prism{j}_{k}", **kw)
        else:
            body.add_geom(name=f"obj{j}_{k}", size=list(part.size) + [0] * (3 - len(part.size)), **kw)


def build_scene(cfg, root: Path, objects: list[LibObject], textured: bool = False) -> mujoco.MjModel:
    """Compile the scene model.

    Args:
        cfg: `env` config node.
        root: directory holding the fetched assets (`third_party/`).
        objects: the library objects of the split; object j becomes free body `obj{j}`.
        textured: add the YCB textures (render-only model: same bodies, joints and geoms, so a physics qpos fits it).

    Returns:
        The compiled model: robot joints first (6 arm, 22 hand), then one free joint per object.
    """
    spec = robot_spec(cfg, root)
    spec.option.timestep = cfg.timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC  # elliptic cones + high impratio: grasps slip less
    spec.option.impratio = 10.0
    spec.visual.global_.offwidth, spec.visual.global_.offheight = cfg.render.width, cfg.render.height
    add_table(spec, cfg)
    for j, obj in enumerate(objects):
        add_object(spec, j, obj, root, textured)
    return spec.compile()
