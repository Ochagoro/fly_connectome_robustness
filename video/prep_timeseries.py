"""Turn recorded spikes into per-frame signals for the video.

For each condition: MN9 rate (both MN9s, exponential filter) -> proboscis extension,
number of neurons active in the last 20 ms -> body jitter (runaway), and a
per-neuron activity trace for the brain panel (saved as sparse frames).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from sim import load_connectome  # noqa: E402
from neurons import Populations  # noqa: E402

FPS = 30
OUT = HERE / "build"
OUT.mkdir(exist_ok=True)

con = load_connectome("783")
pops = Populations(con)
MN9 = pops.readout("MN9")


def signals(name, trial, sim_ms, video_s, tau_mn9=40.0, tau_glow=15.0):
    d = np.load(HERE.parent / f"results/spikes/{name}_trial{trial}.npz")
    step, idx = d["step"], d["idx"]
    n_frames = int(round(video_s * FPS))
    ms_per_frame = sim_ms / n_frames
    t_ms = step * 0.1
    frame_of = np.minimum((t_ms / ms_per_frame).astype(int), 10 ** 9)
    keep = frame_of < n_frames
    frame_of, idx_k, t_k = frame_of[keep], idx[keep], t_ms[keep]
    stim = set(d["stim"].tolist())
    # MN9 rate
    is_mn9 = np.isin(idx_k, MN9)
    mn9_frames = np.bincount(frame_of[is_mn9], minlength=n_frames).astype(float)
    rate = np.zeros(n_frames)
    dec = np.exp(-ms_per_frame / tau_mn9)
    r = 0.0
    for f in range(n_frames):
        r = r * dec + mn9_frames[f] / len(MN9) * 1000.0 / tau_mn9
        rate[f] = r
    # active neurons in the last 20 ms
    win = max(1, int(round(20.0 / ms_per_frame)))
    act = np.zeros(n_frames)
    order = np.argsort(frame_of, kind="stable")
    fo, io = frame_of[order], idx_k[order]
    bounds = np.searchsorted(fo, np.arange(n_frames + 1))
    for f in range(n_frames):
        lo = bounds[max(0, f - win + 1)]
        hi = bounds[f + 1]
        act[f] = len(np.unique(io[lo:hi]))
    ext = np.clip(rate / 45.0, 0, 1)
    jit = np.clip((act - 600) / 2500.0, 0, 1)
    # cumulative MN9 spikes ("drinking commands")
    cum = np.cumsum(mn9_frames)
    t_video = np.arange(n_frames) / FPS
    pd.DataFrame(dict(t=t_video, extension=ext, jitter=jit)).to_csv(OUT / f"{name}_{trial}_{video_s}s.csv", index=False)
    # brain glow traces, stored sparsely per frame
    glow_dec = np.exp(-ms_per_frame / tau_glow)
    tr = np.zeros(con.n, np.float32)
    frames_idx, frames_val, frames_ptr = [], [], [0]
    nonstim = np.ones(con.n, bool)
    nonstim[list(stim)] = False
    for f in range(n_frames):
        tr *= glow_dec
        j = io[bounds[f]:bounds[f + 1]]
        if j.size:
            np.add.at(tr, j, 1.0)
        nz = np.flatnonzero((tr > 0.02) & nonstim)
        frames_idx.append(nz.astype(np.int32))
        frames_val.append(tr[nz].astype(np.float16))
        frames_ptr.append(frames_ptr[-1] + nz.size)
    np.savez_compressed(OUT / f"{name}_{trial}_{video_s}s_glow.npz",
                        idx=np.concatenate(frames_idx), val=np.concatenate(frames_val),
                        ptr=np.array(frames_ptr), rate=rate, act=act, cum=cum,
                        ms_per_frame=ms_per_frame, stim=np.array(sorted(stim)))
    print(name, trial, "frames", n_frames, "max MN9 rate %.1f" % rate.max(), "max active %d" % act.max(),
          "MN9 spikes", int(cum[-1]))


if __name__ == "__main__":
    # round 1: as published; round 2: corrected right hemisphere
    signals("sugar_left_intact", 0, 600.0, 15)
    signals("sugar_right_intact", 0, 600.0, 15)
    signals("sugar_left_intact", 0, 600.0, 12)
    signals("sugar_right_rescued", 0, 600.0, 12)
