"""Reconstruction-error models applied to a connectome edge list.

Each function returns a new Connectome (and optionally per-edge weight
scales) drawn from a seeded random generator. Levels are chosen so that 0
returns the original connectome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sim import Connectome

# Shiu et al. treat GABA and glutamate as inhibitory, everything else as excitatory
NT_SIGN = {"acetylcholine": 1, "gaba": -1, "glutamate": -1, "dopamine": 1,
           "serotonin": 1, "octopamine": 1}


def _rng(seed):
    return np.random.default_rng(seed)


def _drop_zero(con: Connectome, count: np.ndarray, **kw) -> Connectome:
    keep = count > 0
    return con.copy(pre=con.pre[keep], post=con.post[keep], count=count[keep], **kw)


def synapse_deletion(con: Connectome, p: float, seed: int) -> Connectome:
    """False negatives: each synapse is missed independently with probability p."""
    if p <= 0:
        return con
    count = _rng(seed).binomial(con.count, 1.0 - p)
    return _drop_zero(con, count)


def _beta_fraction(rng, n, mean, kappa=4.0):
    a, b = mean * kappa, (1 - mean) * kappa
    return rng.beta(a, b, size=n)


def input_truncation(con: Connectome, p: float, seed: int) -> Connectome:
    """Missing dendritic branches: each neuron loses a random fraction (mean p)
    of all its input synapses, so losses are correlated within a neuron."""
    if p <= 0:
        return con
    rng = _rng(seed)
    f = _beta_fraction(rng, con.n, p)
    count = rng.binomial(con.count, 1.0 - f[con.post])
    return _drop_zero(con, count)


def output_truncation(con: Connectome, p: float, seed: int) -> Connectome:
    """Missing axonal branches: each neuron loses a random fraction (mean p) of its outputs."""
    if p <= 0:
        return con
    rng = _rng(seed)
    f = _beta_fraction(rng, con.n, p)
    count = rng.binomial(con.count, 1.0 - f[con.pre])
    return _drop_zero(con, count)


def false_synapses(con: Connectome, q: float, seed: int) -> Connectome:
    """False positives: add q x (total synapses) spurious synapses. Partners are
    drawn in proportion to each neuron's existing output and input synapse counts."""
    if q <= 0:
        return con
    rng = _rng(seed)
    total = int(con.count.sum())
    n_add = rng.poisson(q * total)
    out_w = np.bincount(con.pre, weights=con.count, minlength=con.n)
    in_w = np.bincount(con.post, weights=con.count, minlength=con.n)
    pre_new = rng.choice(con.n, size=n_add, p=out_w / out_w.sum())
    post_new = rng.choice(con.n, size=n_add, p=in_w / in_w.sum())
    ok = pre_new != post_new
    pre = np.concatenate([con.pre, pre_new[ok]])
    post = np.concatenate([con.post, post_new[ok]])
    cnt = np.concatenate([con.count, np.ones(ok.sum(), np.int64)])
    key = pre * con.n + post
    uk, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv, weights=cnt).astype(np.int64)
    return con.copy(pre=uk // con.n, post=uk % con.n, count=cnt)


def sign_flip(con: Connectome, r: float, seed: int) -> Connectome:
    """Neurotransmitter errors: a random fraction r of neurons with outputs switch
    between excitatory and inhibitory."""
    if r <= 0:
        return con
    rng = _rng(seed)
    has_out = np.zeros(con.n, bool)
    has_out[con.pre] = True
    flip = has_out & (rng.random(con.n) < r)
    sign = con.sign.copy()
    sign[flip] *= -1
    return con.copy(sign=sign)


def nt_resample(con: Connectome, ann: pd.DataFrame, scale: float, seed: int) -> Connectome:
    """Calibrated neurotransmitter errors: each neuron keeps its predicted transmitter
    with probability 1 - scale*(1 - confidence); otherwise a different transmitter is
    drawn in proportion to the transmitters' overall frequency. scale = 1 uses the
    classifier's own confidence as its error rate."""
    if scale <= 0:
        return con
    rng = _rng(seed)
    a = ann.reindex(con.ids)
    nt = a.top_nt.fillna("").to_numpy()
    conf = a.top_nt_conf.fillna(1.0).to_numpy()
    names = list(NT_SIGN)
    freq = pd.Series(nt).value_counts().reindex(names).fillna(0).to_numpy(float)
    err = rng.random(con.n) < np.clip(scale * (1 - conf), 0, 1)
    sign = con.sign.copy()
    for i in np.flatnonzero(err & (nt != "")):
        w = freq.copy()
        w[names.index(nt[i])] = 0
        new = names[rng.choice(len(names), p=w / w.sum())]
        sign[i] = NT_SIGN[new]
    return con.copy(sign=sign)


def glutamate_excitatory(con: Connectome, ann: pd.DataFrame) -> Connectome:
    """Systematic sign assumption: treat glutamatergic neurons as excitatory."""
    a = ann.reindex(con.ids)
    sign = con.sign.copy()
    sign[(a.top_nt == "glutamate").to_numpy()] = 1
    return con.copy(sign=sign)


def weight_noise(con: Connectome, sigma: float, seed: int):
    """Synapse count is an imperfect proxy for strength: multiply each connection's
    weight by log-normal noise with mean 1 (so the average weight is unchanged)."""
    if sigma <= 0:
        return con, None
    scale = _rng(seed).lognormal(-0.5 * sigma ** 2, sigma, size=len(con.count))
    return con, scale


def min_count(con: Connectome, k: int) -> Connectome:
    """Analysis choice: drop connections with fewer than k synapses."""
    if k <= 1:
        return con
    return _drop_zero(con, np.where(con.count >= k, con.count, 0))


ERROR_MODELS = {
    "deletion":     dict(fn=synapse_deletion, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
    "input_trunc":  dict(fn=input_truncation, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    "output_trunc": dict(fn=output_truncation, levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    "false_pos":    dict(fn=false_synapses, levels=[0.05, 0.1, 0.2, 0.4, 0.8]),
    "sign_flip":    dict(fn=sign_flip, levels=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2]),
    "weight_noise": dict(fn=weight_noise, levels=[0.25, 0.5, 1.0, 1.5, 2.0]),
}


def al_serotonin_fix(con: Connectome, ann: pd.DataFrame) -> Connectome:
    """Antennal-lobe local neurons predicted to be serotonergic become inhibitory.
    The only serotonergic neuron known to innervate the antennal lobe is CSD, which is
    not a local neuron, so serotonin predictions for local neurons are likely errors."""
    a = ann.reindex(con.ids)
    sign = con.sign.copy()
    sign[((a.cell_class == "ALLN") & (a.top_nt == "serotonin")).to_numpy()] = -1
    return con.copy(sign=sign)


def type_consensus(con: Connectome, ann: pd.DataFrame) -> Connectome:
    """Give every neuron of a cell type the same sign: the confidence-weighted majority
    of its members' predicted transmitters, pooled over both hemispheres."""
    a = ann.reindex(con.ids)
    nt = a.top_nt.fillna("")
    s = nt.map(NT_SIGN).fillna(0).to_numpy() * a.top_nt_conf.fillna(0).to_numpy()
    df = pd.DataFrame({"ct": a.cell_type.to_numpy(), "s": s})
    vote = df.groupby("ct").s.sum()
    sign = con.sign.copy()
    typed = df.ct.notna().to_numpy()
    v = df.ct[typed].map(vote).to_numpy()
    has_out = np.zeros(con.n, bool)
    has_out[con.pre] = True
    new = np.where(v > 0, 1, np.where(v < 0, -1, sign[typed]))
    sign[typed] = np.where(has_out[typed], new, sign[typed])
    return con.copy(sign=sign)
