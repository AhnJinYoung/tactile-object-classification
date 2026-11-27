"""CPU rendering of one world (EGL offscreen) and an mp4 recorder with a small text/bar overlay.

Rendering never touches the simulation backend's state: the world's qpos is copied into a private MjData.
Set `MUJOCO_GL=egl` before importing mujoco on a headless machine.
"""

from pathlib import Path

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from tacfind.envs.bin_env import BinEnv

TARGET_RGBA = (0.2, 0.9, 0.3, 0.25)


class WorldRenderer:
    """Renders world i of a `BinEnv` from the config camera with the textured objects; a translucent green sphere
    marks the target, red dots the tactile contact points.

    Args:
        env: the environment.
    """

    def __init__(self, env: BinEnv):
        rc = env.cfg.render
        self.env, self.model = env, env.textured_model()
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, rc.height, rc.width)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:], self.cam.distance = rc.camera.lookat, rc.camera.distance
        self.cam.azimuth, self.cam.elevation = rc.camera.azimuth, rc.camera.elevation

    def __call__(self, i: int) -> np.ndarray:
        """RGB frame (H, W, 3) uint8 of world i."""
        m, d, st = self.model, self.data, self.env.world_state(i)
        d.qpos[:] = st["qpos"]
        mujoco.mj_forward(m, d)
        self.renderer.update_scene(d, self.cam)
        scn, eye = self.renderer.scene, np.eye(3).ravel()
        target = int(st["active_body"][st["target_slot"]])
        radius = float(self.env.clouds[int(st["target_lib"]), :, :3].norm(dim=-1).max()) + 0.01
        marks = [(d.xpos[target], radius, TARGET_RGBA)]
        marks += [(p, 0.006, (1, 0.1, 0.1, 1)) for p in st["pts"][st["pts_valid"]]]
        for pos, size, rgba in marks:
            if scn.ngeom < scn.maxgeom:
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE, [size, 0, 0], pos, eye, rgba)
                scn.ngeom += 1
        return self.renderer.render()

    def close(self) -> None:
        """Free the GL context."""
        self.renderer.close()


def overlay(frame: np.ndarray, values: dict[str, float], text: str = "") -> np.ndarray:
    """Draw named scalars in [0, 1] as horizontal bars (plus a text line) in the top-left corner of a frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    for k, (name, v) in enumerate(values.items()):
        y = 8 + 16 * k
        draw.rectangle([8, y, 108, y + 10], outline=(255, 255, 255))
        draw.rectangle([8, y, 8 + int(100 * float(np.clip(v, 0, 1))), y + 10], fill=(80, 200, 120))
        draw.text((114, y - 1), f"{name} {float(v):.2f}", fill=(255, 255, 255))
    if text:
        draw.text((8, 8 + 16 * len(values)), text, fill=(255, 255, 255))
    return np.asarray(img)


class VideoRecorder:
    """Collects frames and writes an mp4.

    Args:
        path: output file (.mp4); parent directories are created.
        fps: frames per second (the env control rate gives real time).
    """

    def __init__(self, path: Path, fps: float):
        self.path, self.fps, self.frames = Path(path), fps, []

    def add(self, frame: np.ndarray) -> None:
        """Append one RGB frame."""
        self.frames.append(frame)

    def save(self) -> Path:
        """Write the video and return its path."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(self.path, self.frames, fps=self.fps, macro_block_size=1)
        return self.path
