"""Stimulus and readout populations for the benchmark predictions (FlyWire v783).

Populations are defined by cell type and hemisphere so that each benchmark can
be run separately from the left and the right side of the same brain.
The cell-type compositions follow the neuron sets of Shiu et al. (2024).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sim import DATA, Connectome

SIDES = ("left", "right")

STIM_TYPES = {
    "sugar":  ["LB3b", "LB3c"],                      # sugar / sugar-low-salt GRNs
    "water":  ["LB3a"],                              # water GRNs
    "bitter": ["LB1a", "LB1b", "LB1c", "LB1d"],      # bitter GRNs
    "ir94e":  ["LB1e", "LB2a", "LB2b", "LB2c"],      # Ir94e-like GRNs (Shiu et al. set)
    "jo":     None,                                  # Johnston's organ C and E neurons
}

READOUT_TYPES = {
    "MN9":  "CB0701",    # proboscis motor neuron 9
    "aDN1": "DNg62",     # antennal grooming descending neuron 1
    "aDN2": "DNge078",   # antennal grooming descending neuron 2
    "aBN1": "SAD093",    # antennal grooming brain neuron 1
}

# benchmark -> (stimulus groups as (population, rate Hz), readout, kind)
BENCHMARKS = {
    "sugar":  dict(stim=[("sugar", 100.0)], readout="MN9", kind="activation"),
    "water":  dict(stim=[("water", 200.0)], readout="MN9", kind="activation"),
    "bitter": dict(stim=[("sugar", 100.0), ("bitter", 100.0)], readout="MN9", kind="suppression",
                   ref="sugar"),
    "ir94e":  dict(stim=[("sugar", 100.0), ("ir94e", 100.0)], readout="MN9", kind="suppression",
                   ref="sugar"),
    "groom":  dict(stim=[("jo", 100.0)], readout="aDN1", kind="activation"),
}


def load_annotations() -> pd.DataFrame:
    cols = ["root_id", "super_class", "cell_class", "cell_sub_class", "cell_type",
            "hemibrain_type", "side", "top_nt", "top_nt_conf", "pos_x", "pos_y", "pos_z",
            "soma_x", "soma_y", "soma_z", "flow"]
    return pd.read_csv(DATA / "neuron_annotations.tsv", sep="\t", usecols=cols, low_memory=False)


class Populations:
    def __init__(self, con: Connectome, ann: pd.DataFrame | None = None):
        self.con = con
        ann = load_annotations() if ann is None else ann
        ann = ann.set_index("root_id").reindex(con.ids)
        self.ann = ann
        self.idx_of = pd.Series(np.arange(con.n), index=con.ids)

    def _sel(self, mask) -> np.ndarray:
        return np.flatnonzero(np.asarray(mask, bool))

    def stim(self, name: str, side: str) -> np.ndarray:
        a = self.ann
        if name == "jo":
            ct = a.cell_type.fillna("")
            m = (ct.str.startswith("JO-C") | ct.str.startswith("JO-E")) & (a.side == side)
        else:
            m = a.cell_type.isin(STIM_TYPES[name]) & (a.side == side)
        return self._sel(m)

    def readout(self, name: str, side: str | None = None) -> np.ndarray:
        a = self.ann
        m = a.cell_type == READOUT_TYPES[name]
        if side is not None:
            m &= a.side == side
        return self._sel(m)

    def benchmark_stim(self, bench: str, side: str):
        return [(self.stim(pop, side), rate) for pop, rate in BENCHMARKS[bench]["stim"]]


def mirror(side: str) -> str:
    return "right" if side == "left" else "left"
