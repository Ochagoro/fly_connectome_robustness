"""Event-driven GPU implementation of the Shiu et al. (2024) whole-brain LIF model.

The network equations, constants and update order follow model.py in
github.com/philshiu/Drosophila_brain_model (Brian 2 version):

    dv/dt = (v_0 - v + g) / t_mbr      (unless refractory)
    dg/dt = -g / tau                    (unless refractory)
    spike: v > v_th  ->  v = v_rst, g = 0, refractory for t_rfc
    presynaptic spike -> g_post += w after a delay t_dly
    w = (signed synapse count) * w_syn
    stimulated neurons receive Poisson input that adds w_syn * f_poi to v
    (and have no refractory period).

Many trials run in parallel as a batch (B x N state). Synaptic input is
delivered event by event: only the outgoing connections of neurons that
spiked are touched, which keeps a 1 s trial of the 138,639-neuron brain at a
few seconds of GPU time.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

PARAMS = dict(
    v_0=-52.0,    # mV
    v_rst=-52.0,  # mV
    v_th=-45.0,   # mV
    t_mbr=20.0,   # ms
    tau=5.0,      # ms
    t_rfc=2.2,    # ms
    t_dly=1.8,    # ms
    w_syn=0.275,  # mV per synapse
    f_poi=250.0,  # Poisson input weight = w_syn * f_poi
    dt=0.1,       # ms
    t_run=1000.0, # ms
)


@dataclass
class Connectome:
    """Edge list of a connectome. Indices are 0..n-1."""
    ids: np.ndarray            # FlyWire root id per index
    pre: np.ndarray            # int64
    post: np.ndarray           # int64
    count: np.ndarray          # synapse count (>=0)
    sign: np.ndarray           # +1 / -1 per neuron (presynaptic sign)
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.ids)

    def index_of(self, root_ids) -> np.ndarray:
        lut = pd.Series(np.arange(self.n), index=self.ids)
        root_ids = [r for r in root_ids if r in lut.index]
        return lut.loc[root_ids].to_numpy()

    def copy(self, **kw) -> "Connectome":
        d = dict(ids=self.ids, pre=self.pre, post=self.post, count=self.count,
                 sign=self.sign, meta=dict(self.meta))
        d.update(kw)
        return Connectome(**d)


def load_connectome(version: str = "783") -> Connectome:
    base = DATA / "shiu_model"
    if version == "783":
        comp, con = base / "Completeness_783.csv", base / "Connectivity_783.parquet"
    elif version == "630":
        comp = base / "2023_03_23_completeness_630_final.csv"
        con = base / "2023_03_23_connectivity_630_final.parquet"
    else:
        raise ValueError(version)
    ids = pd.read_csv(comp, index_col=0).index.to_numpy(np.int64)
    df = pd.read_parquet(con, columns=["Presynaptic_Index", "Postsynaptic_Index",
                                        "Connectivity", "Excitatory"])
    pre = df["Presynaptic_Index"].to_numpy(np.int64)
    post = df["Postsynaptic_Index"].to_numpy(np.int64)
    count = df["Connectivity"].to_numpy(np.int64)
    exc = df["Excitatory"].to_numpy(np.int64)
    sign = np.ones(len(ids), np.int64)
    sign[pre] = exc            # one sign per presynaptic neuron (Dale's law)
    return Connectome(ids=ids, pre=pre, post=post, count=count, sign=sign,
                      meta=dict(version=version))


class Simulator:
    """Batched, event-driven simulator for one connectome."""

    def __init__(self, con: Connectome, params: dict | None = None,
                 device: str = "cuda", weight_scale: np.ndarray | None = None):
        self.p = dict(PARAMS)
        if params:
            self.p.update(params)
        self.dev = torch.device(device)
        self.n = con.n
        p = self.p
        w = con.count.astype(np.float32) * con.sign[con.pre].astype(np.float32) * p["w_syn"]
        if weight_scale is not None:
            w = w * weight_scale.astype(np.float32)
        keep = w != 0
        pre = torch.as_tensor(con.pre[keep], device=self.dev)
        post = torch.as_tensor(con.post[keep], device=self.dev)
        w = torch.as_tensor(w[keep], device=self.dev)
        order = torch.argsort(pre, stable=True)
        self.rows = post[order].contiguous()
        self.vals = w[order].contiguous()
        deg = torch.bincount(pre, minlength=self.n)
        self.col_ptr = torch.zeros(self.n + 1, dtype=torch.int64, device=self.dev)
        self.col_ptr[1:] = torch.cumsum(deg, 0)
        self.n_edges = int(self.rows.numel())

        dt = p["dt"]
        self.eg = math.exp(-dt / p["tau"])
        self.em = math.exp(-dt / p["t_mbr"])
        # exact solution of the linear (v, g) system over one step
        self.cgv = (p["tau"] / (p["tau"] - p["t_mbr"])) * (self.eg - self.em)
        self.delay = int(round(p["t_dly"] / dt))
        self.n_rfc = int(round(p["t_rfc"] / dt))
        self.n_steps = int(round(p["t_run"] / dt))
        self.w_poi = p["w_syn"] * p["f_poi"]

    @torch.no_grad()
    def run(self, stim: list[list[tuple[np.ndarray, float]]],
            silence: list[np.ndarray] | None = None, seed: int = 0,
            record_idx: np.ndarray | None = None, record_all: bool = False,
            t_run: float | None = None):
        """Run len(stim) trials in parallel.

        stim[b] is a list of (neuron indices, Poisson rate in Hz) for trial b.
        silence[b] lists neurons whose output is removed in trial b.
        Returns spike counts (B x N, int32) and optionally spike events.
        """
        B, N, dev, p = len(stim), self.n, self.dev, self.p
        n_steps = self.n_steps if t_run is None else int(round(t_run / p["dt"]))
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))

        v = torch.full((B, N), p["v_0"], device=dev)
        g = torch.zeros((B, N), device=dev)
        refr = torch.zeros((B, N), dtype=torch.int16, device=dev)
        counts = torch.zeros((B, N), dtype=torch.int32, device=dev)
        rfc_steps = torch.full((B, N), self.n_rfc, dtype=torch.int16, device=dev)

        # Poisson targets (flat indices) and their per-step probabilities
        flat, prob = [], []
        for b, groups in enumerate(stim):
            for idx, rate in groups:
                idx = torch.as_tensor(np.asarray(idx, np.int64), device=dev)
                flat.append(b * N + idx)
                prob.append(torch.full((idx.numel(),), rate * p["dt"] * 1e-3, device=dev))
        if flat:
            poi_flat = torch.cat(flat)
            poi_prob = torch.cat(prob)
            rfc_steps.view(-1)[poi_flat] = 0
        else:
            poi_flat = torch.zeros(0, dtype=torch.int64, device=dev)
            poi_prob = torch.zeros(0, device=dev)

        sil = None
        if silence is not None and any(len(s) for s in silence):
            sil = torch.zeros((B, N), dtype=torch.bool, device=dev)
            for b, s in enumerate(silence):
                if len(s):
                    sil[b, torch.as_tensor(np.asarray(s, np.int64), device=dev)] = True

        rec_mask = None
        if record_idx is not None:
            rec_mask = torch.zeros(N, dtype=torch.bool, device=dev)
            rec_mask[torch.as_tensor(np.asarray(record_idx, np.int64), device=dev)] = True
        events = []

        queue: deque = deque([torch.zeros(0, dtype=torch.int64, device=dev)] * self.delay)
        vflat, gflat = v.view(-1), g.view(-1)
        col_ptr, rows, vals = self.col_ptr, self.rows, self.vals
        v0, vth, vrst = p["v_0"], p["v_th"], p["v_rst"]

        for t in range(n_steps):
            # 1) integrate non-refractory neurons
            active = refr <= 0
            v_new = v0 + (v - v0) * self.em + self.cgv * g
            g_new = g * self.eg
            v = torch.where(active, v_new, v)
            g = torch.where(active, g_new, g)
            refr = refr - 1
            # 2) threshold
            spk = (v > vth) & active
            # 3) synaptic events that were emitted `delay` steps ago
            vflat, gflat = v.view(-1), g.view(-1)
            arr = queue.popleft()
            if arr.numel():
                b_idx = torch.div(arr, N, rounding_mode="floor")
                j = arr - b_idx * N
                start = col_ptr[j]
                length = col_ptr[j + 1] - start
                total = int(length.sum())
                if total:
                    seg = torch.repeat_interleave(torch.arange(arr.numel(), device=dev), length,
                                                  output_size=total)
                    offs = torch.arange(total, device=dev) - torch.repeat_interleave(
                        torch.cumsum(length, 0) - length, length, output_size=total)
                    e = start[seg] + offs
                    gflat.index_add_(0, b_idx[seg] * N + rows[e], vals[e])
                # Brian 2 discards synaptic input that arrives while a neuron is
                # refractory (g stays at its reset value of 0); checked against
                # Brian 2 voltage traces in validation_brian2/micro_test.py
                g = torch.where(active, g, torch.zeros_like(g))
                vflat, gflat = v.view(-1), g.view(-1)
            # Poisson inputs act on v directly
            if poi_flat.numel():
                hit = torch.rand(poi_flat.numel(), device=dev, generator=gen) < poi_prob
                vflat.index_add_(0, poi_flat, hit.to(v.dtype) * self.w_poi)
            # 4) reset
            v = torch.where(spk, torch.full_like(v, vrst), v)
            g = torch.where(spk, torch.zeros_like(g), g)
            refr = torch.where(spk, rfc_steps, refr)
            counts += spk.to(torch.int32)
            sflat = torch.nonzero(spk.view(-1)).squeeze(1)
            if record_all or rec_mask is not None:
                if sflat.numel():
                    jj = sflat % N
                    keep = rec_mask[jj] if rec_mask is not None else torch.ones_like(jj, dtype=torch.bool)
                    if keep.any():
                        events.append((t, sflat[keep].cpu()))
            if sil is not None and sflat.numel():
                sflat = sflat[~sil.view(-1)[sflat]]
            queue.append(sflat)

        out = dict(counts=counts.cpu().numpy(), t_run=n_steps * p["dt"])
        if events:
            out["events"] = events
        return out


def rates_from_counts(counts: np.ndarray, t_run_ms: float) -> np.ndarray:
    return counts / (t_run_ms * 1e-3)
