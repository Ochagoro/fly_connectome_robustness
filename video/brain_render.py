"""Render whole-brain spiking activity as glowing points on a frontal view of the fly brain.

Every FlyWire neuron is a point at its annotated position (x, y in 4 nm voxels).
A neuron's brightness is an exponentially decaying trace of its spikes, so a
burst of activity reads as a flash that fades over ~15 ms of simulated time.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

BG = np.array([12, 16, 24], np.float32)        # near-black blue
BRAIN = np.array([70, 82, 100], np.float32)     # faint neuron cloud
GLOW = [(0.0, (255, 196, 64)), (0.5, (255, 120, 40)), (1.0, (255, 255, 230))]


class BrainCanvas:
    def __init__(self, pos_xy: np.ndarray, width: int = 640, height: int = 400, margin: int = 12):
        x, y = pos_xy[:, 0].astype(np.float64), pos_xy[:, 1].astype(np.float64)
        ok = np.isfinite(x) & np.isfinite(y)
        x0, x1 = np.nanpercentile(x[ok], [0.2, 99.8])
        y0, y1 = np.nanpercentile(y[ok], [0.2, 99.8])
        s = min((width - 2 * margin) / (x1 - x0), (height - 2 * margin) / (y1 - y0))
        ox = (width - s * (x1 - x0)) / 2
        oy = (height - s * (y1 - y0)) / 2
        px = np.clip(((x - x0) * s + ox).round(), 0, width - 1)
        py = np.clip(((y - y0) * s + oy).round(), 0, height - 1)
        px[~ok], py[~ok] = -1, -1
        self.px, self.py = px.astype(np.int64), py.astype(np.int64)
        self.ok = ok
        self.w, self.h = width, height
        dens = np.zeros((height, width), np.float32)
        np.add.at(dens, (self.py[ok], self.px[ok]), 1.0)
        dens = gaussian_filter(dens, 1.2)
        dens = np.log1p(dens) / np.log1p(dens).max()
        self.base = BG[None, None, :] + dens[..., None] * (BRAIN - BG)[None, None, :]

    def frame(self, intensity: np.ndarray, gain: float = 1.0) -> np.ndarray:
        """intensity: per-neuron activity (>=0). Returns an RGB uint8 image."""
        act = np.flatnonzero((intensity > 1e-3) & self.ok)
        img = self.base.copy()
        if act.size:
            buf = np.zeros((self.h, self.w), np.float32)
            np.add.at(buf, (self.py[act], self.px[act]), intensity[act])
            glow = gaussian_filter(buf, 1.6) * 6.0 + gaussian_filter(buf, 5.0) * 10.0
            a = np.clip(glow * gain, 0, 1)[..., None]
            col = np.array(GLOW[1][1], np.float32)[None, None, :] * (1 - a) + \
                np.array(GLOW[2][1], np.float32)[None, None, :] * a
            img = img * (1 - a) + col * a
        return np.clip(img, 0, 255).astype(np.uint8)


def traces_from_events(events, n_neurons: int, n_steps: int, frame_steps: int,
                       tau_steps: float = 150.0, trial: int = 0):
    """events: list of (step, flat_idx tensor) from Simulator.run(record_all=True) for B=1.
    Yields a per-neuron intensity vector for every frame."""
    by_step = {}
    for t, fl in events:
        j = fl.numpy() % n_neurons
        by_step[t] = j
    tr = np.zeros(n_neurons, np.float32)
    decay = np.exp(-frame_steps / tau_steps)
    for f in range(n_steps // frame_steps):
        tr *= decay
        for t in range(f * frame_steps, (f + 1) * frame_steps):
            j = by_step.get(t)
            if j is not None:
                np.add.at(tr, j, 1.0)
        yield tr.copy()


def to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)
