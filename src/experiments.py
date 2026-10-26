"""Main experiments. Run as: python experiments.py <name> [...]

  mirror    benchmarks from the left and the right hemisphere across stimulus rates
  version   FlyWire v630 vs v783 with the original Shiu et al. sugar neurons
  errors    error-injection dose-response for every error model
  targeted  deletion restricted to the outputs of the stimulated sugar neurons
  screen    mirror test for every sensory cell type
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

import perturb as P
from neurons import Populations, SIDES, mirror
from sim import ROOT, DATA, Connectome, Simulator, load_connectome

RES = ROOT / "results"
RES.mkdir(exist_ok=True)
RUNAWAY = 2000          # more active neurons than this = runaway
READ = [("MN9", "left"), ("MN9", "right"), ("aDN1", "left"), ("aDN1", "right"),
        ("aDN2", "left"), ("aDN2", "right"), ("aBN1", "left"), ("aBN1", "right")]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def serotonin_fixed(con: Connectome, ann: pd.DataFrame) -> Connectome:
    """Blunt control: every neuron predicted to be serotonergic becomes inhibitory."""
    sign = con.sign.copy()
    sign[(ann.reindex(con.ids).top_nt == "serotonin").to_numpy()] = -1
    return con.copy(sign=sign)


def corrected(con: Connectome, ann: pd.DataFrame) -> Connectome:
    """Cell-type consensus signs plus inhibitory antennal-lobe local neurons."""
    return P.al_serotonin_fix(P.type_consensus(con, ann), ann)


def run_conditions(sim: Simulator, pops: Populations, conds: list[dict], n_trials: int,
                   seed: int, max_batch: int = 96, keep_rates: bool = False):
    """conds: dicts with 'stim' (list of (idx, rate)) plus metadata. Returns a DataFrame
    with readout rates and the number of active neurons, and optionally mean rate vectors."""
    rows, rates = [], []
    per = max(1, max_batch // n_trials)
    for k in range(0, len(conds), per):
        chunk = conds[k:k + per]
        stim = [c["stim"] for c in chunk for _ in range(n_trials)]
        out = sim.run(stim, seed=seed + k)
        r = out["counts"].reshape(len(chunk), n_trials, -1).astype(np.float32).mean(1) / (out["t_run"] * 1e-3)
        for ci, c in enumerate(chunk):
            row = {kk: vv for kk, vv in c.items() if kk != "stim"}
            row["n_active"] = int((r[ci] > 0).sum())
            for nm, sd in READ:
                row[f"{nm}_{sd}"] = float(r[ci, pops.readout(nm, sd)].mean())
            rows.append(row)
            if keep_rates:
                rates.append(r[ci].astype(np.float16))
    df = pd.DataFrame(rows)
    return (df, np.stack(rates)) if keep_rates else df


def bench_conds(pops: Populations, side: str, rate: float, jo_rate: float | None = None):
    s, b, i, j = (pops.stim(n, side) for n in ("sugar", "bitter", "ir94e", "jo"))
    return [
        dict(cond="sugar", side=side, rate=rate, stim=[(s, rate)]),
        dict(cond="sugar+bitter", side=side, rate=rate, stim=[(s, rate), (b, rate)]),
        dict(cond="sugar+ir94e", side=side, rate=rate, stim=[(s, rate), (i, rate)]),
        dict(cond="jo", side=side, rate=jo_rate or rate, stim=[(j, jo_rate or rate)]),
    ]


# ---------------------------------------------------------------- experiments
def exp_mirror():
    con = load_connectome("783")
    pops = Populations(con)
    out = []
    for label, c in [("intact", con), ("serotonin_inhibitory", serotonin_fixed(con, pops.ann))]:
        sim = Simulator(c)
        conds = [cd for rate in [25, 50, 75, 100, 150, 200, 300] for side in SIDES
                 for cd in bench_conds(pops, side, float(rate))]
        df = run_conditions(sim, pops, conds, n_trials=8, seed=100)
        df["connectome"] = label
        out.append(df)
        log("mirror", label, "done")
    pd.concat(out).to_csv(RES / "mirror_benchmarks.csv", index=False)


def exp_version():
    ids = json.load(open(DATA / "shiu_id_lists_v630.json"))["neu_sugar"]
    res = {}
    for v in ["630", "783"]:
        con = load_connectome(v)
        sim = Simulator(con)
        idx = con.index_of(ids)
        rows = []
        for rate in [50., 100., 150., 200.]:
            o = sim.run([[(idx, rate)]] * 30, seed=int(rate))
            r = o["counts"].mean(0)
            res[(v, rate)] = pd.Series(r, index=con.ids)
        log("version", v, "stimulated", len(idx))
    df = pd.DataFrame({f"v{v}_{int(r)}Hz": s for (v, r), s in res.items()})
    df.to_parquet(RES / "version_rates.parquet")


def instances():
    for name, spec in P.ERROR_MODELS.items():
        for lv in spec["levels"]:
            for seed in range(3):
                yield name, lv, seed
    for lv in [0.25, 0.5, 1.0, 2.0]:
        for seed in range(3):
            yield "nt_resample", lv, seed
    for k in [2, 3, 5, 10]:
        yield "min_count", k, 0
    yield "glutamate_excitatory", 1, 0


def build(con, ann, name, lv, seed):
    scale = None
    if name == "weight_noise":
        c, scale = P.weight_noise(con, lv, seed)
    elif name == "nt_resample":
        c = P.nt_resample(con, ann, lv, seed)
    elif name == "min_count":
        c = P.min_count(con, int(lv))
    elif name == "glutamate_excitatory":
        c = P.glutamate_excitatory(con, ann)
    else:
        c = P.ERROR_MODELS[name]["fn"](con, lv, seed)
    return c, scale


def exp_errors(which: str = "all"):
    con = load_connectome("783")
    pops = Populations(con)
    ann = pops.ann
    outdir = RES / "errors"
    outdir.mkdir(exist_ok=True)
    conds = bench_conds(pops, "left", 100.0, jo_rate=150.0)
    todo = [("intact", 0.0, s) for s in range(3)] + list(instances())
    rows = []
    for name, lv, seed in todo:
        if which != "all" and name != which and name != "intact":
            continue
        fn = outdir / f"{name}_{lv}_{seed}.npz"
        c, scale = (con, None) if name == "intact" else build(con, ann, name, lv, seed)
        sim = Simulator(c, weight_scale=scale)
        df, rates = run_conditions(sim, pops, conds, n_trials=8, seed=1000 + seed, keep_rates=True)
        df["model"], df["level"], df["seed"] = name, lv, seed
        df["n_edges"], df["n_syn"] = len(c.count), int(c.count.sum())
        rows.append(df)
        np.savez_compressed(fn, rates=rates)
        log("errors", name, lv, seed, "MN9", round(df.MN9_right.iloc[0], 1),
            "active", df.n_active.tolist())
        pd.concat(rows).to_csv(RES / f"errors_{which}.csv", index=False)


def exp_targeted():
    """Delete a fraction of the output synapses of the stimulated left sugar neurons only.
    The right sugar neurons have about half the output synapses of the left ones."""
    con = load_connectome("783")
    pops = Populations(con)
    L = pops.stim("sugar", "left")
    rows = []
    for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for seed in range(3):
            rng = np.random.default_rng(seed)
            cnt = con.count.copy()
            m = np.isin(con.pre, L)
            cnt[m] = rng.binomial(cnt[m], 1 - frac)
            keep = cnt > 0
            c = con.copy(pre=con.pre[keep], post=con.post[keep], count=cnt[keep])
            for label, cc in [("intact_signs", c), ("al_fix", P.al_serotonin_fix(c, pops.ann))]:
                sim = Simulator(cc)
                conds = [dict(cond="sugar", side="left", rate=float(r), stim=[(L, float(r))])
                         for r in [100, 200, 300]]
                df = run_conditions(sim, pops, conds, n_trials=8, seed=seed)
                df["frac_deleted"], df["seed"], df["connectome"] = frac, seed, label
                rows.append(df)
        log("targeted", frac)
        pd.concat(rows).to_csv(RES / "targeted_sugar_deletion.csv", index=False)


def sensory_types(pops: Populations, min_n: int = 3):
    a = pops.ann
    s = a[a.super_class.isin(["sensory"]) & a.cell_type.notna()]
    g = s.groupby(["cell_type", "side"]).size().unstack(fill_value=0)
    g = g[(g.get("left", 0) >= min_n) & (g.get("right", 0) >= min_n)]
    return g


def type_vectors(rates: np.ndarray, ann: pd.DataFrame, stim_side: str):
    """Mean rate per (cell type, ipsi/contra) relative to the stimulated side."""
    ct = ann.cell_type.to_numpy()
    side = ann.side.to_numpy()
    rel = np.where(side == stim_side, "ipsi", np.where(side == mirror(stim_side), "contra", "mid"))
    key = pd.Series(ct.astype(object)) + "|" + pd.Series(rel)
    ok = ~pd.isna(ct)
    df = pd.DataFrame(rates[:, ok].T.astype(np.float32))
    df["key"] = key[ok].to_numpy()
    return df.groupby("key").mean().T  # conditions x keys


def exp_screen():
    con = load_connectome("783")
    pops = Populations(con)
    a = pops.ann
    types = sensory_types(pops)
    log("screen types", len(types))
    outdir = RES / "screen"
    outdir.mkdir(exist_ok=True)
    out_syn = np.bincount(con.pre, weights=con.count, minlength=con.n)
    feats = []
    for t in types.index:
        f = dict(cell_type=t, cell_class=a[a.cell_type == t].cell_class.iloc[0])
        for sd in SIDES:
            idx = np.flatnonzero((a.cell_type == t).to_numpy() & (a.side == sd).to_numpy())
            f[f"n_{sd}"] = len(idx)
            f[f"syn_{sd}"] = float(out_syn[idx].sum())
        feats.append(f)
    pd.DataFrame(feats).to_csv(RES / "screen_features.csv", index=False)
    for label, c in [("intact", con), ("al_fix", P.al_serotonin_fix(con, a)), ("corrected", corrected(con, a))]:
        sim = Simulator(c)
        conds = []
        for t in types.index:
            for sd in SIDES:
                idx = np.flatnonzero((a.cell_type == t).to_numpy() & (a.side == sd).to_numpy())
                conds.append(dict(cell_type=t, side=sd, rate=100.0, stim=[(idx, 100.0)]))
        df, rates = run_conditions(sim, pops, conds, n_trials=4, seed=7, keep_rates=True)
        df["connectome"] = label
        df.to_csv(RES / f"screen_{label}.csv", index=False)
        # mirrored cell-type vectors
        vecs = []
        for i, cd in enumerate(conds):
            v = type_vectors(rates[i:i + 1], a, cd["side"])
            v.index = [i]
            vecs.append(v)
        V = pd.concat(vecs).fillna(0)
        V.to_parquet(outdir / f"type_vectors_{label}.parquet")
        np.save(outdir / f"rates_{label}.npy", rates)
        log("screen", label, "done")


def compensation_scale(con: Connectome, pops: Populations, inputs=True, readouts=True):
    """Per-edge weight scales that undo the right-hemisphere synapse deficits:
    outputs of each right stimulus population are scaled to the left per-neuron mean,
    and inputs to the right aBN1 are scaled to the left aBN1 total."""
    scale = np.ones(len(con.count), np.float32)
    out_syn = np.bincount(con.pre, weights=con.count, minlength=con.n)
    in_syn = np.bincount(con.post, weights=con.count, minlength=con.n)
    info = {}
    if inputs:
        for k in ["sugar", "bitter", "ir94e", "jo"]:
            L, R = pops.stim(k, "left"), pops.stim(k, "right")
            f = out_syn[L].mean() / out_syn[R].mean()
            if f > 1:
                scale[np.isin(con.pre, R)] *= f
            info[f"in_{k}"] = float(f)
    if readouts:
        for k in ["aBN1", "aDN1", "aDN2", "MN9"]:
            L, R = pops.readout(k, "left"), pops.readout(k, "right")
            f = in_syn[L].sum() / in_syn[R].sum()
            if f > 1:
                scale[np.isin(con.post, R)] *= f
            info[f"out_{k}"] = float(f)
    return scale, info


def exp_rescue():
    con = load_connectome("783")
    pops = Populations(con)
    al = P.al_serotonin_fix(con, pops.ann)
    s_in, info_in = compensation_scale(con, pops, inputs=True, readouts=False)
    s_all, info_all = compensation_scale(con, pops, inputs=True, readouts=True)
    log("compensation", info_all)
    variants = [("intact", con, None), ("al_fix", al, None),
                ("al_fix+inputs", al, s_in), ("al_fix+inputs+readouts", al, s_all)]
    rows = []
    for label, c, sc in variants:
        sim = Simulator(c, weight_scale=sc)
        conds = [cd for s in SIDES for cd in bench_conds(pops, s, 100.0, jo_rate=150.0)]
        df = run_conditions(sim, pops, conds, n_trials=10, seed=21)
        df["variant"] = label
        rows.append(df)
        log("rescue", label)
    out = pd.concat(rows)
    out.to_csv(RES / "rescue.csv", index=False)
    json.dump(info_all, open(RES / "compensation_factors.json", "w"), indent=1)


def exp_reliability():
    """Within-side reproducibility of the screen (noise ceiling): rerun both sides with new seeds."""
    con = load_connectome("783")
    pops = Populations(con)
    a = pops.ann
    types = sensory_types(pops)
    outdir = RES / "screen"
    sim = Simulator(P.al_serotonin_fix(con, a))
    conds = []
    for t in types.index:
        for sd in SIDES:
            idx = np.flatnonzero((a.cell_type == t).to_numpy() & (a.side == sd).to_numpy())
            conds.append(dict(cell_type=t, side=sd, rate=100.0, stim=[(idx, 100.0)]))
    df, rates = run_conditions(sim, pops, conds, n_trials=4, seed=9001, keep_rates=True)
    df.to_csv(RES / "screen_al_fix_repeat.csv", index=False)
    vecs = []
    for i, cd in enumerate(conds):
        v = type_vectors(rates[i:i + 1], a, cd["side"])
        v.index = [i]
        vecs.append(v)
    pd.concat(vecs).fillna(0).to_parquet(outdir / "type_vectors_al_fix_repeat.parquet")
    log("reliability done")


if __name__ == "__main__":
    for name in sys.argv[1:]:
        log("start", name)
        {"mirror": exp_mirror, "version": exp_version, "errors": exp_errors, "rescue": exp_rescue,
         "targeted": exp_targeted, "screen": exp_screen,
         "reliability": exp_reliability}[name]()
        log("end", name)
