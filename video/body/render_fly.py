#!/usr/bin/env python
"""Kinematic NeuroMechFly (flygym 2.x) renderer for proboscis-extension videos.

The fly body is posed purely kinematically: every frame we write joint angles
into ``qpos``, call ``mj_forward`` and render offscreen. No physics is stepped.

Input CSV columns
-----------------
t          time in seconds (monotonic; resampled to the output fps)
extension  0..1  proboscis extension amount (0 = folded, 1 = fully extended)
jitter     0..1  amplitude of random twitching of legs / antennae / head
                 (used to depict runaway brain activity)

Usage
-----
CLI:
    MUJOCO_GL=egl .venv/bin/python render_fly.py in.csv out.mp4 \
        --width 640 --height 480 --fps 30 --camera side_head [--still frame.png]

Python:
    from render_fly import render
    render("in.csv", "out.mp4", width=640, height=480, fps=30, camera="side_head")

Units of the model are millimetres; the fly is ~2.5 mm long.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

# Offscreen GL backend must be chosen before mujoco is imported.
os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ["MUJOCO_GL"] in ("egl", "osmesa"):
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import numpy as np  # noqa: E402
import mujoco as mj  # noqa: E402

DEG = np.pi / 180.0

# --------------------------------------------------------------------------
# Proboscis kinematics (degrees, added on top of the neutral pose).
# extension e in [0, 1] is mapped through a smoothstep onto these angles.
# --------------------------------------------------------------------------
PROBOSCIS_JOINTS_DEG = {
    # joint name                         retracted  extended
    "c_head-c_rostrum-pitch": (0.0, -50.0),  # rostrum swings forward/down
    "c_rostrum-c_haustellum-pitch": (0.0, 125.0),  # haustellum+labellum unfold
}
# flygym's default proboscis hinge points sit at the segment-origin corners, which
# opens visible gaps when rotated by large angles. We move the hinge pivots
# (body-frame, mm) to points inside the proximal end of each segment so the
# proboscis stays connected while extending.
PROBOSCIS_PIVOTS = {
    "c_head-c_rostrum": (-0.12, 0.0, 0.12),
    "c_rostrum-c_haustellum": (0.06, 0.0, -0.06),
}

# Joints that twitch under "jitter" with their amplitude (deg) at jitter=1.
# Leg joints are perturbed around the resting pose; tarsi follow loosely.
_LEG_JITTER = {
    "c_thorax-{s}_coxa-pitch": 14.0,
    "c_thorax-{s}_coxa-roll": 8.0,
    "{s}_coxa-{s}_trochanterfemur-pitch": 18.0,
    "{s}_trochanterfemur-{s}_tibia-pitch": 22.0,
    "{s}_tibia-{s}_tarsus1-pitch": 12.0,
}
_OTHER_JITTER = {
    "c_thorax-c_head-pitch": 7.0,
    "c_thorax-c_head-yaw": 7.0,
    "c_thorax-c_head-roll": 6.0,
    "c_head-l_pedicel-pitch": 20.0,
    "c_head-l_pedicel-yaw": 20.0,
    "c_head-r_pedicel-pitch": 20.0,
    "c_head-r_pedicel-yaw": 20.0,
    "c_thorax-l_wing-yaw": 6.0,
    "c_thorax-r_wing-yaw": 6.0,
    "c_thorax-l_wing-roll": 6.0,
    "c_thorax-r_wing-roll": 6.0,
    "c_thorax-c_abdomen12-pitch": 4.0,
    "c_head-c_rostrum-pitch": 6.0,
}
LEGS = ["lf", "lm", "lh", "rf", "rm", "rh"]
DROPLET_RADIUS = 0.16  # mm


def _jitter_table() -> dict[str, float]:
    tab = dict(_OTHER_JITTER)
    for s in LEGS:
        for k, v in _LEG_JITTER.items():
            tab[k.format(s=s)] = v
    return tab


# Camera presets: (lookat offset relative to head [mm], distance, azimuth, elevation)
# MuJoCo free camera: azimuth 90 = pure side view of the fly's right side (head
# to the right of the image), 180 = looking straight at the face. A pure side view
# hides the proboscis behind the near front leg, hence the slight frontal rotation.
CAMERAS = {
    "side_head": dict(lookat=(0.25, 0.0, -0.6), distance=3.0, azimuth=120.0, elevation=-3.0),
    "three_quarter": dict(lookat=(0.2, 0.0, -0.6), distance=3.2, azimuth=140.0, elevation=-4.0),
    "front": dict(lookat=(0.2, 0.0, -0.6), distance=3.2, azimuth=160.0, elevation=-4.0),
    "side_full": dict(lookat=(-0.3, 0.0, -0.6), distance=4.6, azimuth=105.0, elevation=-6.0),
}


@dataclass
class FlyScene:
    model: mj.MjModel
    data: mj.MjData
    neutral_qpos: np.ndarray
    qadr: dict[str, int]
    droplet_geom: int | None
    head_body: int
    haustellum_body: int


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_scene(droplet: bool = True, floor_rgb=(0.86, 0.85, 0.82)) -> FlyScene:
    """Compose NeuroMechFly + floor + lights + optional sugar droplet."""
    warnings.filterwarnings("ignore")
    from flygym.compose import NeuroMechFly, KinematicPosePreset
    from flygym.anatomy import Skeleton, JointPreset, AxisOrder

    fly = NeuroMechFly()
    skeleton = Skeleton(
        axis_order=AxisOrder.YAW_PITCH_ROLL, joint_preset=JointPreset.ALL_POSSIBLE
    )
    fly.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
    fly.colorize()
    spec = fly.mjcf_root
    spec.compiler.fusestatic = False
    neutral_qpos = np.array(spec.keys[0].qpos, dtype=float)

    # ---- place the fly so all six tarsal tips rest on z=0 ------------------
    m0 = spec.copy().compile()
    d0 = mj.MjData(m0)
    d0.qpos[:] = neutral_qpos
    mj.mj_forward(m0, d0)
    tips = np.array([_tarsus_tip(m0, d0, s) for s in LEGS])
    # least squares z = a + b*x  -> pitch the fly so that the feet plane is level
    A = np.c_[np.ones(len(tips)), tips[:, 0]]
    a, b = np.linalg.lstsq(A, tips[:, 2], rcond=None)[0]
    pitch = np.arctan(b)  # rotate by -pitch about y  (positive b: front is higher)
    thorax = spec.body("c_thorax")
    pivot = np.array([0.0, 0.0, a])
    R = _rot_y(pitch)  # rotating about +y by +pitch lowers +x points
    old_pos = np.array(thorax.pos, dtype=float)
    new_pos = R @ (old_pos - pivot) + pivot
    q_rot = np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])
    new_quat = _quat_mul(q_rot, np.array(thorax.quat, dtype=float))
    # after rotation, recompute the lowest tip and drop the fly to the floor
    tips_rot = (R @ (tips - pivot).T).T + pivot
    dz = -tips_rot[:, 2].min() + 0.012  # tiny clearance avoids z-fighting
    thorax.pos = new_pos + np.array([0.0, 0.0, dz])
    thorax.quat = new_quat

    # ---- environment -------------------------------------------------------
    spec.add_texture(
        name="skybox_bg",
        type=mj.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mj.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=(0.97, 0.97, 0.97),
        rgb2=(0.86, 0.87, 0.89),
        width=256,
        height=256,
    )
    spec.add_material(name="floor_mat", rgba=(*floor_rgb, 1.0), specular=0.05, shininess=0.1, reflectance=0.0)
    spec.worldbody.add_geom(
        name="floor",
        type=mj.mjtGeom.mjGEOM_PLANE,
        size=(40.0, 40.0, 0.1),
        pos=(0.0, 0.0, 0.0),
        material="floor_mat",
        contype=0,
        conaffinity=0,
    )
    spec.worldbody.add_light(
        name="key", pos=(1.0, -3.0, 8.0), dir=(-0.15, 0.35, -1.0), diffuse=(0.55, 0.55, 0.55),
        specular=(0.15, 0.15, 0.15), castshadow=True, type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    spec.worldbody.add_light(
        name="fill", pos=(-3.0, 4.0, 6.0), dir=(0.3, -0.5, -1.0), diffuse=(0.22, 0.22, 0.22),
        specular=(0.0, 0.0, 0.0), castshadow=False, type=mj.mjtLightType.mjLIGHT_DIRECTIONAL,
    )
    spec.visual.headlight.ambient = (0.38, 0.38, 0.38)
    spec.visual.headlight.diffuse = (0.30, 0.30, 0.30)
    spec.visual.headlight.specular = (0.05, 0.05, 0.05)
    spec.visual.quality.shadowsize = 4096
    spec.visual.quality.offsamples = 8

    # slightly paler proboscis so it reads against the legs
    spec.add_material(name="proboscis_mat", rgba=(0.80, 0.62, 0.36, 1.0), specular=0.25, shininess=0.3)
    for gname in ("c_rostrum", "c_haustellum"):
        spec.geom(gname).material = "proboscis_mat"

    if droplet:
        # sugar-water drop presented on a glass capillary (classic PER assay look)
        spec.add_material(name="droplet_mat", rgba=(0.62, 0.84, 1.0, 0.5), specular=0.9, shininess=0.95)
        spec.add_material(name="glass_mat", rgba=(0.75, 0.80, 0.85, 0.35), specular=0.8, shininess=0.9)
        dbody = spec.worldbody.add_body(name="droplet_body", mocap=True, pos=(0.0, 0.0, 0.0))
        dbody.add_geom(
            name="droplet", type=mj.mjtGeom.mjGEOM_SPHERE, size=(DROPLET_RADIUS, 0.0, 0.0),
            material="droplet_mat", contype=0, conaffinity=0,
        )
        cap_dir = np.array([0.8, 0.3, 0.55]) / np.linalg.norm([0.8, 0.3, 0.55])
        dbody.add_geom(
            name="capillary", type=mj.mjtGeom.mjGEOM_CAPSULE, size=(0.06, 0.0, 0.0),
            fromto=(*(cap_dir * DROPLET_RADIUS * 0.6), *(cap_dir * 6.0)),
            material="glass_mat", contype=0, conaffinity=0,
        )

    model = spec.compile()
    for prefix, piv in PROBOSCIS_PIVOTS.items():
        for ax in ("yaw", "pitch", "roll"):
            model.jnt_pos[model.joint(f"{prefix}-{ax}").id] = piv
    data = mj.MjData(model)
    qadr = {model.joint(j).name: int(model.jnt_qposadr[j]) for j in range(model.njnt)}
    scene = FlyScene(
        model=model,
        data=data,
        neutral_qpos=neutral_qpos,
        qadr=qadr,
        droplet_geom=model.geom("droplet").id if droplet else None,
        head_body=model.body("c_head").id,
        haustellum_body=model.body("c_haustellum").id,
    )
    if droplet:
        _place_droplet(scene)
    return scene


def _place_droplet(scene: FlyScene) -> None:
    """Place the capillary-held droplet so the fully-extended labellum touches it."""
    m, d = scene.model, scene.data
    d.qpos[:] = pose_qpos(scene, extension=1.0)
    mj.mj_forward(m, d)
    # The drop hangs from the capillary just in front of / below the labellum tip
    tip_dir = np.array([0.6, 0.0, -0.8])
    tip = _geom_extreme_point(m, d, "c_haustellum", direction=tip_dir)
    d.mocap_pos[0] = tip + np.array([0.75, 0.0, -0.35]) * DROPLET_RADIUS


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _mesh_world_vertices(m, d, geom_name):
    g = m.geom(geom_name).id
    mesh_id = m.geom_dataid[g]
    a = m.mesh_vertadr[mesh_id]
    n = m.mesh_vertnum[mesh_id]
    v = m.mesh_vert[a:a + n]
    R = d.geom_xmat[g].reshape(3, 3)
    return v @ R.T + d.geom_xpos[g]


def _tarsus_tip(m, d, leg):
    v = _mesh_world_vertices(m, d, f"{leg}_tarsus5")
    return v[np.argmin(v[:, 2])]


def _geom_extreme_point(m, d, geom_name, direction):
    v = _mesh_world_vertices(m, d, geom_name)
    return v[np.argmax(v @ direction)]


def pose_qpos(scene: FlyScene, extension: float, jitter_offsets: dict[str, float] | None = None) -> np.ndarray:
    """Return qpos for a given proboscis extension and jitter offsets (radians)."""
    q = scene.neutral_qpos.copy()
    e = float(_smoothstep(extension))
    for name, (a0, a1) in PROBOSCIS_JOINTS_DEG.items():
        q[scene.qadr[name]] += (a0 + (a1 - a0) * e) * DEG
    if jitter_offsets:
        for name, off in jitter_offsets.items():
            q[scene.qadr[name]] += off
    return q


class JitterGenerator:
    """Smooth-ish random twitches: band-limited noise + sparse fast jerks."""

    def __init__(self, fps: float, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.table = _jitter_table()
        self.names = list(self.table)
        self.amp = np.array([self.table[n] for n in self.names]) * DEG
        self.state = np.zeros(len(self.names))
        self.target = np.zeros(len(self.names))
        self.fps = fps

    def step(self, level: float) -> dict[str, float]:
        if level <= 1e-6:
            self.state *= 0.5
            return dict(zip(self.names, self.state))
        # new random targets frequently (twitch), occasionally large jerks
        resample = self.rng.random(len(self.names)) < 0.35 + 0.4 * level
        self.target[resample] = self.rng.uniform(-1, 1, resample.sum())
        jerk = self.rng.random(len(self.names)) < 0.05 * level
        self.target[jerk] = np.sign(self.rng.uniform(-1, 1, jerk.sum())) * 1.3
        alpha = 0.55
        self.state = (1 - alpha) * self.state + alpha * self.target
        return dict(zip(self.names, self.state * self.amp * level))


def _make_camera(scene: FlyScene, camera: str | dict) -> mj.MjvCamera:
    spec = CAMERAS[camera] if isinstance(camera, str) else camera
    cam = mj.MjvCamera()
    cam.type = mj.mjtCamera.mjCAMERA_FREE
    mj.mj_forward(scene.model, scene.data)
    head = scene.data.xpos[scene.head_body].copy()
    cam.lookat[:] = head + np.array(spec["lookat"])
    cam.distance = spec["distance"]
    cam.azimuth = spec["azimuth"]
    cam.elevation = spec["elevation"]
    return cam


def _read_csv(path):
    import csv

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    t = np.array([float(r["t"]) for r in rows])
    ext = np.array([float(r.get("extension", 0) or 0) for r in rows])
    jit = np.array([float(r.get("jitter", 0) or 0) for r in rows])
    order = np.argsort(t)
    return t[order], np.clip(ext[order], 0, 1), np.clip(jit[order], 0, 1)


class FlyRenderer:
    """Reusable renderer (build the scene once, render many frames)."""

    def __init__(self, width=640, height=480, camera="side_head", droplet=True, seed=0, fps=30):
        self.scene = build_scene(droplet=droplet)
        self.renderer = mj.Renderer(self.scene.model, height=height, width=width)
        self.scene.data.qpos[:] = self.scene.neutral_qpos
        self.cam = _make_camera(self.scene, camera)
        self.jitter = JitterGenerator(fps=fps, seed=seed)
        self.opt = mj.MjvOption()

    def frame(self, extension: float, jitter: float, offsets=None) -> np.ndarray:
        sc = self.scene
        offs = self.jitter.step(jitter) if offsets is None else offsets
        sc.data.qpos[:] = pose_qpos(sc, extension, offs)
        mj.mj_forward(sc.model, sc.data)
        self.renderer.update_scene(sc.data, camera=self.cam, scene_option=self.opt)
        return self.renderer.render()

    def close(self):
        self.renderer.close()


def render(
    timeseries_csv,
    out_mp4,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    camera: str | dict = "side_head",
    droplet: bool = True,
    seed: int = 0,
    still_png: str | None = None,
    still_at: str = "auto",
    verbose: bool = True,
) -> dict:
    """Render a CSV time series (t, extension, jitter) to an mp4.

    still_png: optional path; saves one frame (``still_at``: 'auto' = frame with
    max extension+jitter, or a float time in seconds).
    Returns timing stats.
    """
    import imageio.v2 as imageio

    t, ext, jit = _read_csv(timeseries_csv)
    n_frames = max(1, int(round((t[-1] - t[0]) * fps)) + 1)
    tf = t[0] + np.arange(n_frames) / fps
    ef = np.interp(tf, t, ext)
    jf = np.interp(tf, t, jit)

    t0 = time.perf_counter()
    fr = FlyRenderer(width=width, height=height, camera=camera, droplet=droplet, seed=seed, fps=fps)
    t_build = time.perf_counter() - t0

    if still_png is not None:
        if still_at == "auto":
            still_idx = int(np.argmax(ef + jf))
        else:
            still_idx = int(np.clip(round((float(still_at) - t[0]) * fps), 0, n_frames - 1))
    else:
        still_idx = -1

    Path(out_mp4).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(out_mp4), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=8
    )
    t1 = time.perf_counter()
    for i in range(n_frames):
        img = fr.frame(ef[i], jf[i])
        writer.append_data(img)
        if i == still_idx:
            from PIL import Image

            Image.fromarray(img).save(still_png)
    writer.close()
    t_render = time.perf_counter() - t1
    fr.close()
    stats = dict(
        frames=n_frames,
        build_s=round(t_build, 3),
        render_s=round(t_render, 3),
        ms_per_frame=round(1000 * t_render / n_frames, 2),
    )
    if verbose:
        print(f"[render_fly] {out_mp4}: {stats}", file=sys.stderr)
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="CSV with columns t, extension, jitter")
    p.add_argument("out", help="output .mp4")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--camera", default="side_head", choices=list(CAMERAS))
    p.add_argument("--no-droplet", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--still", default=None, help="also save a PNG still")
    p.add_argument("--still-at", default="auto", help="'auto' or time in seconds")
    a = p.parse_args(argv)
    render(
        a.csv, a.out, width=a.width, height=a.height, fps=a.fps, camera=a.camera,
        droplet=not a.no_droplet, seed=a.seed, still_png=a.still, still_at=a.still_at,
    )


if __name__ == "__main__":
    main()
