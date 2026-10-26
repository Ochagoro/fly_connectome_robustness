"""Main and Extended Data figures. Run: python figures.py [fig1 fig2 fig3 fig4 ed1 ...]"""
from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import style as S
from neurons import Populations, SIDES
from sim import ROOT, load_connectome

sys.path.insert(0, str(ROOT / "video"))
from brain_render import BrainCanvas  # noqa: E402

S.apply()
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
_cache = {}


def ctx():
    if "con" not in _cache:
        con = load_connectome("783")
        _cache["con"], _cache["pops"] = con, Populations(con)
    return _cache["con"], _cache["pops"]


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    fig.savefig(FIG / f"{name}.svg")
    plt.close(fig)
    print("saved", name)


def side_label(s):
    return {"left": "Left hemisphere", "right": "Right hemisphere"}[s]


def benchmark_table(df: pd.DataFrame, keys=("variant",)) -> pd.DataFrame:
    """Six readouts per (keys, side) from rows with cond in sugar/sugar+bitter/sugar+ir94e/jo."""
    out = []
    keys = [c for c in keys if c in df.columns]
    for gk, g in df.groupby(keys + ["side"]):
        gk = gk if isinstance(gk, tuple) else (gk,)
        side = gk[-1]
        contra = "right" if side == "left" else "left"
        r = {k: v for k, v in zip(keys + ["side"], gk)}
        mn9 = lambda c: g[g.cond == c][["MN9_left", "MN9_right"]].to_numpy().mean() if (g.cond == c).any() else np.nan
        s, b, i = mn9("sugar"), mn9("sugar+bitter"), mn9("sugar+ir94e")
        jo = g[g.cond == "jo"]
        r.update({
            "Sugar → MN9 (Hz)": s,
            "Bitter suppression": 1 - b / s if s > 1 else np.nan,
            "Ir94e suppression": 1 - i / s if s > 1 else np.nan,
            "JO → aBN1, ipsi (Hz)": jo[f"aBN1_{side}"].mean() if len(jo) else np.nan,
            "JO → aDN1, contra (Hz)": jo[f"aDN1_{contra}"].mean() if len(jo) else np.nan,
            "JO → aDN2, contra (Hz)": jo[f"aDN2_{contra}"].mean() if len(jo) else np.nan,
            "n_active_sugar": g[g.cond == "sugar"].n_active.mean(),
        })
        out.append(r)
    return pd.DataFrame(out)


SUCCESS = {"Sugar → MN9 (Hz)": ("gt", 20), "Bitter suppression": ("gt", 0.5),
           "Ir94e suppression": ("gt", 0.5), "JO → aBN1, ipsi (Hz)": ("gt", 20),
           "JO → aDN1, contra (Hz)": ("gt", 10), "JO → aDN2, contra (Hz)": ("gt", 10)}


def passed(row) -> dict:
    out = {}
    for k, (op, th) in SUCCESS.items():
        v = row[k]
        ok = (v > th) if np.isfinite(v) else False
        if k in ("Bitter suppression", "Ir94e suppression") and not row["Sugar → MN9 (Hz)"] > SUCCESS["Sugar → MN9 (Hz)"][1]:
            ok = False
        out[k] = bool(ok)
    return out


# ------------------------------------------------------------------ Figure 1
def brain_snapshot(name, t0_ms=300, t1_ms=700, w=520, h=270):
    con, pops = ctx()
    if "canvas" not in _cache:
        a = pops.ann
        _cache["canvas"] = BrainCanvas(np.stack([a.pos_x.to_numpy(), a.pos_y.to_numpy()], 1), w, h)
    d = np.load(RES / f"spikes/{name}_trial0.npz")
    m = (d["step"] >= t0_ms * 10) & (d["step"] < t1_ms * 10)
    rate = np.bincount(d["idx"][m], minlength=con.n).astype(np.float32) / ((t1_ms - t0_ms) * 1e-3)
    inten = np.log1p(rate) / np.log1p(200)
    inten[d["stim"]] = 0
    return _cache["canvas"].frame(inten * 0.35)


def cumulative_active(name, trials=3):
    curves = []
    for tr in range(trials):
        d = np.load(RES / f"spikes/{name}_trial{tr}.npz")
        stim = set(d["stim"].tolist())
        first = pd.Series(d["step"]).groupby(d["idx"]).min()
        first = first[~first.index.isin(stim)]
        t = np.arange(0, 10001, 10)
        curves.append(np.searchsorted(np.sort(first.to_numpy()), t, side="right"))
    return np.arange(0, 10001, 10) * 0.1, np.array(curves)


def fig1():
    con, pops = ctx()
    mb = pd.read_csv(RES / "mirror_benchmarks.csv")
    mb = mb[mb.connectome == "intact"]
    fig = plt.figure(figsize=(S.DOUBLE, 118 * S.MM))
    gs = fig.add_gridspec(2, 6, height_ratios=[1, 1.05], hspace=0.62, wspace=1.4,
                          left=0.06, right=0.99, top=0.95, bottom=0.08)

    # a: whole-brain activity snapshots
    ax = fig.add_subplot(gs[0, 0:3])
    img = np.concatenate([brain_snapshot("sugar_left_intact"), np.full((270, 8, 3), 255, np.uint8),
                          brain_snapshot("sugar_right_intact")], 1)
    ax.imshow(img)
    ax.set_axis_off()
    ax.text(0.25, -0.06, "Left sugar GRNs, 100 Hz", transform=ax.transAxes, ha="center", va="top", color=S.LEFT, fontsize=6.5)
    ax.text(0.75, -0.06, "Right sugar GRNs, 100 Hz", transform=ax.transAxes, ha="center", va="top", color=S.RIGHT, fontsize=6.5)
    ax.set_title("Mean firing 300–700 ms after stimulus onset (frontal view)", fontsize=6.5, pad=3)
    S.panel_label(ax, "a", x=-0.03, y=1.03)

    # b: MN9 dose-response
    ax = fig.add_subplot(gs[0, 3:5])
    for side, col in [("left", S.LEFT), ("right", S.RIGHT)]:
        d = mb[(mb.cond == "sugar") & (mb.side == side)].sort_values("rate")
        y = d[["MN9_left", "MN9_right"]].mean(axis=1)
        ax.plot(d.rate, y, "-o", color=col, ms=3, label=side_label(side), clip_on=False)
    ax.set_xlabel("Sugar GRN stimulation (Hz)")
    ax.set_ylabel("MN9 firing rate (Hz)")
    ax.set_xlim(0, 310)
    ax.set_ylim(0, 90)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.04))
    S.panel_label(ax, "b")

    # c: number of active neurons
    ax = fig.add_subplot(gs[0, 5])
    for side, col in [("left", S.LEFT), ("right", S.RIGHT)]:
        d = mb[(mb.cond == "sugar") & (mb.side == side)].sort_values("rate")
        ax.plot(d.rate, d.n_active, "-o", color=col, ms=3, clip_on=False)
    ax.set_yscale("log")
    ax.set_ylim(20, 30000)
    ax.set_xlabel("Stimulation (Hz)")
    ax.set_ylabel("Active neurons")
    ax.text(305, 8700 * 1.4, "runaway", color=S.RIGHT, ha="right", fontsize=6)
    S.panel_label(ax, "c", x=-0.6)

    # d: cumulative recruitment over time
    ax = fig.add_subplot(gs[1, 0:2])
    for name, col, lab in [("sugar_left_intact", S.LEFT, "Left"), ("sugar_right_intact", S.RIGHT, "Right")]:
        t, c = cumulative_active(name)
        for k, cc in enumerate(c):
            ax.plot(t, cc, color=col, lw=0.8, label=lab if k == 0 else None)
    ax.set_yscale("log")
    ax.set_ylim(1, 20000)
    ax.set_xlim(0, 600)
    ax.set_xlabel("Time after stimulus onset (ms)")
    ax.set_ylabel("Neurons recruited")
    ax.legend(loc="upper left")
    S.panel_label(ax, "d")

    # e, f: six benchmark predictions, left vs right (activation in Hz, suppression in %)
    rs = pd.read_csv(RES / "rescue.csv")
    bt = benchmark_table(rs[rs.variant == "intact"])
    groups = [("e", ["Sugar → MN9 (Hz)", "JO → aBN1, ipsi (Hz)", "JO → aDN1, contra (Hz)", "JO → aDN2, contra (Hz)"],
               "Firing rate (Hz)", 1, gs[1, 2:5], (0, 80)),
              ("f", ["Bitter suppression", "Ir94e suppression"], "Suppression of MN9 (%)", 100, gs[1, 5], (0, 110))]
    for letter, names, ylab, mult, slot, ylim in groups:
        ax = fig.add_subplot(slot)
        x = np.arange(len(names))
        for k, (side, col) in enumerate([("left", S.LEFT), ("right", S.RIGHT)]):
            row = bt[bt.side == side].iloc[0]
            ok = passed(row)
            vals = [np.nan_to_num(row[n]) * mult for n in names]
            ax.bar(x + (k - 0.5) * 0.36, vals, width=0.34, color=col, label=side_label(side))
            for xi, n, v in zip(x, names, vals):
                ax.text(xi + (k - 0.5) * 0.36, max(v, 0) + ylim[1] * 0.02, "✓" if ok[n] else "×", ha="center",
                        va="bottom", fontsize=6.5, color=S.INK if ok[n] else S.RED, fontfamily="DejaVu Sans")
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" (Hz)", "").replace(", ", "\n").replace(" suppression", "") for n in names],
                           fontsize=5.6)
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        if letter == "e":
            ax.legend(loc="upper right", ncol=2)
            ax.set_title("Benchmark predictions of Shiu et al.", fontsize=6.5, pad=3)
        S.panel_label(ax, letter, x=-0.12 if letter == "e" else -0.55)
    save(fig, "fig1_mirror_test")


# ------------------------------------------------------------------ Figure 2
def per_neuron_outputs():
    con, pops = ctx()
    out_syn = np.bincount(con.pre, weights=con.count, minlength=con.n)
    rows = []
    for k in ["sugar", "bitter", "ir94e", "water", "jo"]:
        for s in SIDES:
            for v in out_syn[pops.stim(k, s)]:
                rows.append(dict(pop=k, side=s, out=v))
    return pd.DataFrame(rows)


def fig2():
    con, pops = ctx()
    a = pops.ann
    fig = plt.figure(figsize=(S.DOUBLE, 112 * S.MM))
    gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.55, left=0.07, right=0.99, top=0.94, bottom=0.1)

    # a: output synapses per stimulated neuron
    ax = fig.add_subplot(gs[0, 0])
    po = per_neuron_outputs()
    labels = {"sugar": "Sugar", "bitter": "Bitter", "ir94e": "Ir94e", "water": "Water", "jo": "JO-C/E"}
    rng = np.random.default_rng(0)
    for i, k in enumerate(labels):
        for j, (s, col) in enumerate([("left", S.LEFT), ("right", S.RIGHT)]):
            v = po[(po["pop"] == k) & (po.side == s)].out.to_numpy()
            xx = i + (j - 0.5) * 0.36 + rng.uniform(-0.09, 0.09, len(v))
            ax.scatter(xx, v, s=2.5, color=col, lw=0, alpha=0.7)
            ax.plot([i + (j - 0.5) * 0.36 - 0.14, i + (j - 0.5) * 0.36 + 0.14], [v.mean()] * 2, color=S.INK, lw=1)
        L = po[(po["pop"] == k) & (po.side == "left")].out.mean()
        R = po[(po["pop"] == k) & (po.side == "right")].out.mean()
        ax.text(i, 760, f"{R / L:.2f}×", ha="center", fontsize=5.5, color=S.INK2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels.values(), rotation=0, fontsize=5.6)
    ax.set_ylabel("Output synapses per neuron")
    ax.set_ylim(0, 800)
    ax.set_title("Right/left ratio of means", fontsize=6, pad=8, color=S.INK2)
    S.panel_label(ax, "a")

    # b: targeted deletion of left sugar-GRN outputs, on an absolute synapse axis
    ax = fig.add_subplot(gs[0, 1])
    td = pd.read_csv(RES / "targeted_sugar_deletion.csv")
    td["MN9"] = td[["MN9_left", "MN9_right"]].mean(axis=1)
    totL = po[(po["pop"] == "sugar") & (po.side == "left")].out.sum()
    totR = po[(po["pop"] == "sugar") & (po.side == "right")].out.sum()
    d = td[(td.connectome == "al_fix") & (td.rate == 100)].groupby("frac_deleted").MN9.agg(["mean", "min", "max"])
    x = (1 - d.index.to_numpy()) * totL / 1000
    ax.plot(x, d["mean"], "-o", color=S.LEFT, ms=2.5, label="Left GRNs, synapses removed")
    ax.fill_between(x, d["min"], d["max"], color=S.LEFT, alpha=0.2, lw=0)
    rs = pd.read_csv(RES / "rescue.csv")
    rr = rs[(rs.variant == "al_fix") & (rs.cond == "sugar") & (rs.side == "right")]
    right_mn9 = rr[["MN9_left", "MN9_right"]].to_numpy().mean()
    ax.plot([totR / 1000], [right_mn9], "D", color=S.RIGHT, ms=3.5, label="Right GRNs (all 24, observed)")
    ax.annotate(f"right: {totR / 1000:.1f}k synapses", xy=(totR / 1000, right_mn9), xytext=(totR / 1000 + 0.6, 18),
                fontsize=5.5, color=S.RIGHT, arrowprops=dict(arrowstyle="-", lw=0.5, color=S.RIGHT))
    ax.set_xlabel("Output synapses of stimulated sugar GRNs (×1,000)")
    ax.set_ylabel("MN9 firing rate (Hz)")
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0, 70)
    ax.legend(loc="upper left", fontsize=5.5)
    S.panel_label(ax, "b")

    # c: readout neuron aBN1 inputs
    ax = fig.add_subplot(gs[0, 2])
    in_syn = np.bincount(con.post, weights=con.count, minlength=con.n)
    names = ["MN9", "aBN1", "aDN1", "aDN2"]
    for j, (s, col) in enumerate([("left", S.LEFT), ("right", S.RIGHT)]):
        vals = [in_syn[pops.readout(n, s)].sum() for n in names]
        ax.bar(np.arange(4) + (j - 0.5) * 0.36, vals, width=0.34, color=col, label=side_label(s))
    ax.set_xticks(range(4))
    ax.set_xticklabels(names)
    ax.set_ylabel("Input synapses of readout neuron")
    ax.legend(loc="upper right", fontsize=5.5)
    ax.annotate("0.25×", xy=(1 + 0.18, in_syn[pops.readout("aBN1", "right")].sum()), xytext=(1.9, 2600),
                fontsize=5.5, color=S.INK2, arrowprops=dict(arrowstyle="-", lw=0.5, color=S.INK2))
    S.panel_label(ax, "c")

    # d: transmitter confidence of the runaway gateway
    ax = fig.add_subplot(gs[1, 0])
    ln = a[a.cell_class == "ALLN"]
    groups = [("Serotonin", ln[ln.top_nt == "serotonin"]), ("GABA", ln[ln.top_nt == "gaba"]),
              ("Glutamate", ln[ln.top_nt == "glutamate"]), ("ACh", ln[ln.top_nt == "acetylcholine"])]
    for i, (nm, g) in enumerate(groups):
        v = g.top_nt_conf.dropna().to_numpy()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=2.5, lw=0, alpha=0.6,
                   color=S.RIGHT if nm == "Serotonin" else S.MUTED)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=S.INK, lw=1)
        ax.text(i, 1.03, f"n = {len(v)}", ha="center", fontsize=5.2, color=S.INK2)
    ax.set_xticks(range(4))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("Transmitter prediction confidence")
    ax.set_ylim(0, 1.08)
    ax.set_title("Antennal-lobe local neurons", fontsize=6, pad=8)
    S.panel_label(ax, "d")

    # e: cumulative recruitment with and without the AL fix
    ax = fig.add_subplot(gs[1, 1])
    for name, col, lab in [("sugar_right_intact", S.RIGHT, "Right, as published"),
                           ("sugar_right_alfix", S.AQUA, "Right, AL sign fix"),
                           ("sugar_left_intact", S.LEFT, "Left, as published")]:
        t, c = cumulative_active(name)
        ax.plot(t, c.mean(0), color=col, label=lab)
    ax.set_yscale("log")
    ax.set_xlim(0, 1000)
    ax.set_ylim(1, 20000)
    ax.set_xlabel("Time after stimulus onset (ms)")
    ax.set_ylabel("Neurons recruited")
    ax.legend(loc="lower right", fontsize=5.3)
    S.panel_label(ax, "e")

    # f: rescue matrix
    ax = fig.add_subplot(gs[1, 2])
    rs = pd.read_csv(RES / "rescue.csv")
    bt = benchmark_table(rs)
    order = [("intact", "left", "Left, as published"), ("intact", "right", "Right, as published"),
             ("al_fix", "right", "+ AL sign fix"), ("al_fix+inputs", "right", "+ sensory outputs"),
             ("al_fix+inputs+readouts", "right", "+ readout inputs")]
    names = list(SUCCESS)
    M = np.zeros((len(order), len(names)))
    for i, (v, s, _) in enumerate(order):
        row = bt[(bt.variant == v) & (bt.side == s)].iloc[0]
        ok = passed(row)
        M[i] = [ok[n] for n in names]
    cmap = LinearSegmentedColormap.from_list("ok", ["#f6dcdc", "#cfe8d6"])
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "✓" if M[i, j] else "×", ha="center", va="center", fontsize=6.5,
                    color=S.INK if M[i, j] else S.RED, fontfamily="DejaVu Sans")
        ax.text(M.shape[1] - 0.4, i, f"{int(M[i].sum())}/6", va="center", fontsize=6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([o[2] for o in order], fontsize=5.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["Sugar", "Bitter", "Ir94e", "aBN1", "aDN1", "aDN2"], fontsize=5.6)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Benchmarks reproduced", fontsize=6, pad=3)
    S.panel_label(ax, "f", x=-0.62)
    save(fig, "fig2_defects")




# ------------------------------------------------------------ Extended Data
def ed1():
    """Validation of the GPU simulator against the original Brian 2 implementation."""
    fig, axes = plt.subplots(1, 4, figsize=(S.DOUBLE, 50 * S.MM), gridspec_kw=dict(wspace=0.55))
    items = [("validation_sugarL_783_150Hz", "Left sugar GRNs, 150 Hz", S.LEFT),
             ("validation_sugar_left_783_100Hz", "Left sugar GRNs, 100 Hz", S.LEFT),
             ("validation_sugar_right_783_100Hz", "Right sugar GRNs, 100 Hz", S.RIGHT)]
    for ax, (f, title, col), letter in zip(axes[:3], items, "abc"):
        x = pd.read_csv(RES / f"{f}.csv", index_col=0)
        ax.scatter(x.brian2, x.gpu, s=2, color=col, lw=0, alpha=0.5, rasterized=True)
        m = max(x.brian2.max(), x.gpu.max()) * 1.05
        ax.plot([0, m], [0, m], color=S.MUTED, lw=0.5, ls="--")
        r = np.corrcoef(x.brian2, x.gpu)[0, 1]
        ax.text(0.05, 0.95, f"r = {r:.3f}\nn = {S.fmt_int(len(x))}", transform=ax.transAxes, va="top", fontsize=5.8)
        ax.set_xlabel("Brian 2 rate (Hz)")
        ax.set_ylabel("GPU rate (Hz)")
        ax.set_title(title, fontsize=6.2)
        ax.set_xlim(0, m)
        ax.set_ylim(0, m)
        S.panel_label(ax, letter, x=-0.42)
    ax = axes[3]
    v = pd.read_parquet(RES / "version_rates.parquet").dropna()
    a, b = v["v630_100Hz"], v["v783_100Hz"]
    mm = (a > 0) | (b > 0)
    ax.scatter(a[mm], b[mm], s=2, color=S.INK2, lw=0, alpha=0.5, rasterized=True)
    ax.plot([0, 200], [0, 200], color=S.MUTED, lw=0.5, ls="--")
    ax.text(0.05, 0.95, f"r = {np.corrcoef(a[mm], b[mm])[0, 1]:.3f}\nn = {mm.sum()}", transform=ax.transAxes,
            va="top", fontsize=5.8)
    ax.set_xlabel("FlyWire v630 rate (Hz)")
    ax.set_ylabel("FlyWire v783 rate (Hz)")
    ax.set_title("Proofreading versions, 100 Hz", fontsize=6.2)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    S.panel_label(ax, "d", x=-0.42)
    save(fig, "ed1_validation")




# ------------------------------------------------------------------ Figure 3
ERR_LABEL = {"deletion": "Missing synapses", "input_trunc": "Truncated dendrites",
             "output_trunc": "Truncated axons", "false_pos": "False synapses",
             "sign_flip": "Sign errors", "weight_noise": "Weight noise"}
ERR_X = {"deletion": "Synapses removed (%)", "input_trunc": "Mean input loss (%)",
         "output_trunc": "Mean output loss (%)", "false_pos": "Spurious synapses added (%)",
         "sign_flip": "Neurons with flipped sign (%)", "weight_noise": "Log-normal σ"}


def error_metrics():
    e = pd.read_csv(RES / "errors_all.csv")
    rows = []
    intact = {c: [] for c in range(4)}
    for (m, lv, sd), g in e.groupby(["model", "level", "seed"]):
        g = g.set_index("cond")
        s = g.loc["sugar", ["MN9_left", "MN9_right"]].mean()
        b = g.loc["sugar+bitter", ["MN9_left", "MN9_right"]].mean()
        i = g.loc["sugar+ir94e", ["MN9_left", "MN9_right"]].mean()
        # suppression is only defined when sugar alone still drives MN9 (> 20 Hz)
        r = dict(model=m, level=lv, seed=sd, sugar=s,
                 bitter=1 - b / s if s > 20 else np.nan, ir94e=1 - i / s if s > 20 else np.nan,
                 aBN1=g.loc["jo", "aBN1_left"], aDN1=g.loc["jo", "aDN1_right"],
                 runaway=bool((g.n_active > 2000).any()), n_active_sugar=g.loc["sugar", "n_active"])
        f = RES / f"errors/{m}_{lv}_{sd}.npz"
        if not f.exists() and float(lv).is_integer():
            f = RES / f"errors/{m}_{int(lv)}_{sd}.npz"
        r["file"] = str(f)
        rows.append(r)
    df = pd.DataFrame(rows)
    # network similarity to the mean intact response (log rates over the union of active neurons)
    I = np.mean([np.load(f)["rates"].astype(np.float32) for f in df[df.model == "intact"].file], axis=0)
    sims = []
    for f in df.file:
        R = np.load(f)["rates"].astype(np.float32)
        cs = []
        for c in range(R.shape[0]):
            m = (I[c] > 0) | (R[c] > 0)
            cs.append(np.corrcoef(np.log1p(I[c][m]), np.log1p(R[c][m]))[0, 1])
        sims.append(np.mean(cs))
    df["similarity"] = sims
    base = df[df.model == "intact"][["sugar", "bitter", "ir94e", "aBN1", "aDN1"]].mean()
    for k in base.index:
        df[f"{k}_rel"] = df[k] / base[k]
    return df, base


BENCH_STYLE = [("sugar_rel", "Sugar → MN9", S.BLUE, "o"), ("bitter_rel", "Bitter suppression", S.ORANGE, "s"),
               ("ir94e_rel", "Ir94e suppression", S.AQUA, "^"), ("aBN1_rel", "JO → aBN1", S.YELLOW, "D"),
               ("aDN1_rel", "JO → aDN1", S.MAGENTA, "v")]


def fig3():
    df, base = error_metrics()
    df.to_csv(RES / "error_metrics.csv", index=False)
    # effective fraction of neurons whose sign changes under calibrated transmitter errors
    nt_x = {0.25: 6.1, 0.5: 12.2, 1.0: 24.3, 2.0: 47.0}
    intact = df[df.model == "intact"]
    panels = [("deletion", "Missing synapses", "Synapses removed (%)", 100, None),
              ("input_trunc", "Truncated dendrites", "Mean input loss (%)", 100, None),
              ("output_trunc", "Truncated axons", "Mean output loss (%)", 100, None),
              ("false_pos", "Spurious synapses", "Spurious synapses added (%)", 100, None),
              ("sign_flip", "Random sign errors", "Neurons with flipped sign (%)", 100, None),
              ("nt_resample", "Transmitter errors (calibrated)", "Neurons with changed sign (%)", None, nt_x),
              ("weight_noise", "Weight noise (mean 1)", "Log-normal σ", 1, None)]
    fig = plt.figure(figsize=(S.DOUBLE, 112 * S.MM))
    gs = fig.add_gridspec(2, 4, hspace=0.78, wspace=0.42, left=0.06, right=0.99, top=0.94, bottom=0.1)
    tol = []
    for k, (m, title, xl, scale, xmap) in enumerate(panels):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        d = pd.concat([intact.assign(level=0.0), df[df.model == m]])
        xs_of = (lambda lv: xmap.get(lv, 0.0)) if xmap else (lambda lv: lv * scale)
        for col, lab, c, mk in BENCH_STYLE:
            g = d.groupby("level")[col].mean()
            if "suppression" in lab:  # undefined once sugar alone fails in every draw
                g = g.dropna()
            x = np.array([xs_of(l) for l in g.index])
            ax.plot(x, g.values, "-", marker=mk, color=c, ms=2.4, lw=0.9, label=lab)
            ys = g.values
            x50 = np.nan
            for j in range(1, len(x)):
                if ys[j - 1] >= 0.5 > ys[j]:
                    x50 = x[j - 1] + (ys[j - 1] - 0.5) / (ys[j - 1] - ys[j]) * (x[j] - x[j - 1])
                    break
            if "suppression" in lab and np.isnan(x50) and len(x) and x.max() < max(xs_of(l) for l in d.level.unique()):
                x50 = -1.0  # undefined: sugar response lost before suppression halved
            tol.append(dict(model=m, title=title, bench=lab, x50=x50, color=c, marker=mk))
        run = d.groupby("level").runaway.mean()
        for lv, fr in run.items():
            if fr > 0:
                ax.text(xs_of(lv), 1.36, "R", ha="center", va="center", fontsize=5.3, color=S.RED,
                        fontweight="bold", alpha=0.35 + 0.65 * fr)
        if m == "nt_resample":
            ax.axvline(6, color=S.MUTED, lw=0.5, ls=":")
            ax.text(7, 0.05, "classifier\nerror ≈ 6%", fontsize=5, color=S.INK2)
        ax.set_ylim(-0.55, 1.45)
        ax.set_yticks([-0.5, 0, 0.5, 1.0])
        ax.axhline(1, color=S.GRID, lw=0.5, zorder=0)
        ax.set_title(title, fontsize=6.5, pad=2)
        ax.set_xlabel(xl)
        if k % 4 == 0:
            ax.set_ylabel("Response (relative to intact)")
        S.panel_label(ax, "abcdefg"[k], x=-0.3 if k % 4 == 0 else -0.2)
        if m == "false_pos":
            ax.legend(loc="lower left", fontsize=5.2, handlelength=1.6, borderaxespad=0.2)
            ax.text(0.98, 0.42, "R = runaway\n(darker = more draws)", transform=ax.transAxes, ha="right",
                    fontsize=5, color=S.RED)
    tol = pd.DataFrame(tol)
    tol.to_csv(RES / "error_tolerance.csv", index=False)
    # h: tolerance summary
    ax = fig.add_subplot(gs[1, 3])
    order = ["sign_flip", "nt_resample", "input_trunc", "output_trunc", "deletion", "false_pos"]
    short = {"sign_flip": "Random sign", "nt_resample": "Transmitter", "input_trunc": "Dendrites",
             "output_trunc": "Axons", "deletion": "Missing syn.", "false_pos": "Spurious syn."}
    off = {"Sugar → MN9": -0.3, "Bitter suppression": -0.15, "Ir94e suppression": 0.0, "JO → aBN1": 0.15,
           "JO → aDN1": 0.3}
    for yi, m in enumerate(order):
        for _, r in tol[tol.model == m].iterrows():
            if r.x50 == -1.0:
                continue
            cens = not np.isfinite(r.x50)
            x = 100 if cens else r.x50
            ax.scatter(x, yi + off[r.bench], s=9, marker=r.marker, facecolor="white" if cens else r.color,
                       edgecolor=r.color, lw=0.7, clip_on=False)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([short[m] for m in order], fontsize=5.6)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(3, 100)
    ax.set_xticks([3, 10, 30, 100])
    ax.set_xticklabels(["3", "10", "30", "≥100"])
    ax.set_xlabel("Error level at 50% loss (%)")
    ax.set_title("Tolerance (open = never)", fontsize=6.5, pad=2)
    S.panel_label(ax, "h", x=-0.5)
    save(fig, "fig3_error_dose_response")


# ------------------------------------------------------------------ Figure 4
def vec_r(a, b):
    m = (a > 0) | (b > 0)
    if m.sum() < 3:
        return np.nan
    return np.corrcoef(np.log1p(a[m]), np.log1p(b[m]))[0, 1]


def screen_table():
    F = pd.read_csv(RES / "screen_features.csv").set_index("cell_type")
    out = {}
    for v in ["intact", "al_fix", "corrected"]:
        df = pd.read_csv(RES / f"screen_{v}.csv")
        V = pd.read_parquet(RES / f"screen/type_vectors_{v}.parquet")
        rows = []
        for t in df.cell_type.unique():
            iL = df.index[(df.cell_type == t) & (df.side == "left")][0]
            iR = df.index[(df.cell_type == t) & (df.side == "right")][0]
            col = "n_active_nonstim" if "n_active_nonstim" in df else "n_active"
            rows.append(dict(cell_type=t, r_LR=vec_r(V.loc[iL].to_numpy(), V.loc[iR].to_numpy()),
                             act_L=df[col][iL], act_R=df[col][iR]))
        R = pd.DataFrame(rows).set_index("cell_type").join(F)
        R["run_L"], R["run_R"] = R.act_L > 2000, R.act_R > 2000
        out[v] = R
    rep = RES / "screen/type_vectors_al_fix_repeat.parquet"
    if rep.exists():
        df = pd.read_csv(RES / "screen_al_fix.csv")
        V = pd.read_parquet(RES / "screen/type_vectors_al_fix.parquet")
        V2 = pd.read_parquet(rep)
        rr = {}
        for t in df.cell_type.unique():
            iL = df.index[(df.cell_type == t) & (df.side == "left")][0]
            iR = df.index[(df.cell_type == t) & (df.side == "right")][0]
            rr[t] = np.nanmean([vec_r(V.loc[iL].to_numpy(), V2.loc[iL].to_numpy()),
                                vec_r(V.loc[iR].to_numpy(), V2.loc[iR].to_numpy())])
        out["al_fix"]["r_rep"] = pd.Series(rr)
    for v, R in out.items():
        R["syn_ratio"] = np.log2(R.syn_right / R.syn_left)
        R["pn_ratio"] = np.log2((R.syn_right / R.n_right) / (R.syn_left / R.n_left))
    return out


CLASS_ORDER = ["olfactory", "mechanosensory", "gustatory", "hygrosensory", "thermosensory", "visual", "unknown_sensory"]
CLASS_LABEL = {"olfactory": "Olfactory", "mechanosensory": "Mechanosensory", "gustatory": "Gustatory",
               "hygrosensory": "Hygrosensory", "thermosensory": "Thermosensory", "visual": "Visual",
               "unknown_sensory": "Unclassified"}


def fig4():
    T = screen_table()
    pd.concat({k: v for k, v in T.items()}).to_csv(RES / "screen_summary.csv")
    fig = plt.figure(figsize=(S.DOUBLE, 60 * S.MM))
    gs = fig.add_gridspec(1, 4, wspace=0.62, left=0.115, right=0.99, top=0.88, bottom=0.21,
                          width_ratios=[1.25, 1, 1, 1])
    # a: runaway incidence by class
    ax = fig.add_subplot(gs[0, 0])
    variants = [("intact", "As published", S.ORANGE), ("al_fix", "AL sign fix (40 neurons)", S.AQUA),
                ("corrected", "AL fix + type-consensus signs", S.BLUE)]
    y = np.arange(len(CLASS_ORDER))
    for k, (v, lab, col) in enumerate(variants):
        R = T[v]
        frac = [((R.cell_class == c) & (R.run_L | R.run_R)).sum() / max((R.cell_class == c).sum(), 1) for c in CLASS_ORDER]
        ax.barh(y + (k - 1) * 0.27, np.array(frac) * 100, height=0.25, color=col, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{CLASS_LABEL[c]} ({(T['intact'].cell_class == c).sum()})" for c in CLASS_ORDER], fontsize=5.8)
    ax.invert_yaxis()
    ax.set_xlabel("Cell types that trigger runaway (%)")
    ax.set_xlim(0, 105)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.71), fontsize=5.2)
    S.panel_label(ax, "a", x=-0.62)
    # b: mirror concordance versus within-side reproducibility
    ax = fig.add_subplot(gs[0, 1])
    R = T["al_fix"].dropna(subset=["r_LR"])
    R = R[~(R.run_L | R.run_R)]
    has_rep = "r_rep" in R
    xcol = "r_rep" if has_rep else "r_LR"
    gust = R.cell_class == "gustatory"
    rng = np.random.default_rng(1)
    for i, (col, lab) in enumerate([("r_rep", "Same side,\nrepeated"), ("r_LR", "Left versus\nright")]):
        v = R[col].to_numpy()
        xx = i + rng.uniform(-0.17, 0.17, len(v))
        ax.scatter(xx[~gust.to_numpy()], v[~gust.to_numpy()], s=4, color=S.MUTED, lw=0, alpha=0.7)
        ax.scatter(xx[gust.to_numpy()], v[gust.to_numpy()], s=6, color=S.ORANGE, lw=0)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=S.INK, lw=1.1)
        ax.text(i + 0.32, np.median(v), f"{np.median(v):.2f}", va="center", fontsize=5.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Same side,\nrepeated", "Left versus\nright"], fontsize=5.8)
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel("Response similarity (r)")
    ax.text(0.02, 0.02, "orange: gustatory types", transform=ax.transAxes, fontsize=5, color=S.ORANGE)
    S.panel_label(ax, "b", x=-0.42)
    # c: discordance versus synapse asymmetry of the stimulated population
    ax = fig.add_subplot(gs[0, 2])
    R["disc"] = (R[xcol] - R.r_LR) if has_rep else (1 - R.r_LR)
    ax.scatter(R.syn_ratio.abs()[~gust], R.disc[~gust], s=5, color=S.MUTED, lw=0, alpha=0.8)
    ax.scatter(R.syn_ratio.abs()[gust], R.disc[gust], s=7, color=S.ORANGE, lw=0)
    from scipy.stats import spearmanr
    rho, p = spearmanr(R.syn_ratio.abs(), R.disc, nan_policy="omit")
    ax.text(0.97, 0.95, f"Spearman ρ = {rho:.2f}\nP = {p:.1g}\nn = {R.disc.notna().sum()}", transform=ax.transAxes, ha="right", va="top", fontsize=5.6)
    ax.set_xlabel("|log$_2$ (right/left output synapses)|")
    ax.set_ylabel("Mirror discordance")
    S.panel_label(ax, "c", x=-0.42)
    # d: fraction of evoked activity that stays on the stimulated side
    ax = fig.add_subplot(gs[0, 3])
    lat = pd.read_csv(RES / "screen_lateral_al_fix.csv")
    runaway_types = T["al_fix"].index[T["al_fix"].run_L | T["al_fix"].run_R]
    lat = lat[~lat.cell_type.isin(runaway_types)]
    p = lat.pivot(index="cell_type", columns="side", values="left_frac").dropna()
    cls = lat.drop_duplicates("cell_type").set_index("cell_type").cls.reindex(p.index)
    xL, yR = p["left"], 1 - p["right"]
    for c, col, lab in [("olfactory", S.BLUE, "Olfactory"), ("mechanosensory", S.AQUA, "Mechanosensory")]:
        m = cls == c
        ax.scatter(xL[m], yR[m], s=6, color=col, lw=0, alpha=0.85, label=lab)
    m = ~cls.isin(["olfactory", "mechanosensory"])
    ax.scatter(xL[m], yR[m], s=6, color=S.MUTED, lw=0, alpha=0.85, label="Other")
    ax.plot([0, 1], [0, 1], color=S.INK2, lw=0.5, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Ipsilateral share, left stimulation")
    ax.set_ylabel("Ipsilateral share, right stimulation")
    ax.legend(loc="upper left", fontsize=4.9)
    S.panel_label(ax, "d", x=-0.42)
    save(fig, "fig4_screen")
    return T




def ed2():
    """Brain-wide similarity, synapse-count thresholds and the glutamate sign assumption."""
    df, base = error_metrics()
    intact = df[df.model == "intact"]
    fig, axes = plt.subplots(1, 3, figsize=(S.DOUBLE, 52 * S.MM), gridspec_kw=dict(wspace=0.5))
    ax = axes[0]
    nt_x = {0.25: 6.1, 0.5: 12.2, 1.0: 24.3, 2.0: 47.0}
    for m, c, mk, lab in [("deletion", S.BLUE, "o", "Missing synapses"), ("input_trunc", S.BLUE, "s", "Dendrites"),
                          ("output_trunc", S.BLUE, "^", "Axons"), ("false_pos", S.AQUA, "D", "False synapses"),
                          ("sign_flip", S.ORANGE, "o", "Random sign"), ("nt_resample", S.ORANGE, "s", "Transmitter")]:
        d = df[df.model == m].groupby("level").similarity.mean()
        x = [nt_x[l] if m == "nt_resample" else l * 100 for l in d.index]
        ax.plot(x, d.values, "-", marker=mk, color=c, ms=2.2, lw=0.8, label=lab,
                mfc="white" if m in ("input_trunc", "output_trunc", "nt_resample") else c)
    ax.set_xscale("log")
    ax.set_xlim(0.4, 100)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Error level (%)")
    ax.set_ylabel("Brain-wide similarity to intact (r)")
    ax.legend(loc="lower left", fontsize=4.8, ncol=2)
    S.panel_label(ax, "a", x=-0.32)
    ax = axes[1]
    d = pd.concat([intact.assign(level=1.0), df[df.model == "min_count"]])
    for col, lab, c, mk in BENCH_STYLE:
        g = d.groupby("level")[col].mean()
        ax.plot(g.index, g.values, "-", marker=mk, color=c, ms=2.4, lw=0.9, label=lab)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10])
    ax.set_xticklabels(["1", "2", "3", "5", "10"])
    ax.set_ylim(-0.05, 1.25)
    ax.set_xlabel("Minimum synapses per connection")
    ax.set_ylabel("Response (relative to intact)")
    ax.legend(loc="lower left", fontsize=4.8)
    S.panel_label(ax, "b", x=-0.32)
    ax = axes[2]
    names = ["Sugar → MN9", "Bitter suppression", "Ir94e suppression", "JO → aBN1", "JO → aDN1"]
    for k, (lab, d) in enumerate([("As published", intact), ("Glutamate excitatory\n(runaway: 30,128 neurons)", df[df.model == "glutamate_excitatory"])]):
        vals = [d[c].mean() for c, *_ in BENCH_STYLE]
        ax.bar(np.arange(5) + (k - 0.5) * 0.36, vals, width=0.34, color=[S.INK2, S.ORANGE][k], label=lab)
    ax.axhline(0, color=S.INK, lw=0.5)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Sugar\n→ MN9", "Bitter\nsupp.", "Ir94e\nsupp.", "JO\n→ aBN1", "JO\n→ aDN1"], fontsize=5.2)
    ax.set_ylabel("Response (relative to intact)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.18), fontsize=5.0, ncol=2)
    S.panel_label(ax, "c", x=-0.32)
    save(fig, "ed2_analysis_choices")


def ed3():
    """Screen details: active neurons from left versus right stimulation for every sensory type."""
    T = screen_table()
    fig, axes = plt.subplots(1, 3, figsize=(S.DOUBLE, 56 * S.MM), gridspec_kw=dict(wspace=0.45))
    for ax, (v, title), letter in zip(axes, [("intact", "As published"), ("al_fix", "AL sign fix"),
                                              ("corrected", "AL fix + type-consensus signs")], "abc"):
        R = T[v]
        for c, col in [("olfactory", S.BLUE), ("mechanosensory", S.AQUA), ("gustatory", S.ORANGE)]:
            m = R.cell_class == c
            ax.scatter(R.act_L[m], R.act_R[m], s=6, color=col, lw=0, alpha=0.85, label=CLASS_LABEL[c])
        m = ~R.cell_class.isin(["olfactory", "mechanosensory", "gustatory"])
        ax.scatter(R.act_L[m], R.act_R[m], s=6, color=S.MUTED, lw=0, alpha=0.85, label="Other")
        ax.plot([10, 20000], [10, 20000], color=S.INK2, lw=0.5, ls="--")
        ax.axhline(2000, color=S.RED, lw=0.4, ls=":")
        ax.axvline(2000, color=S.RED, lw=0.4, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(8, 20000)
        ax.set_ylim(8, 20000)
        ax.set_xlabel("Active neurons, left stimulation")
        ax.set_ylabel("Active neurons, right stimulation")
        ax.set_title(title, fontsize=6.3)
        S.panel_label(ax, letter, x=-0.38)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=4.8, framealpha=0.9, frameon=True, edgecolor="none")
    save(fig, "ed3_screen_active")




def ed4():
    """Rate sensitivity, original neuron sets and the single right-hemisphere gateway neuron."""
    con, pops = ctx()
    a = pops.ann
    fig, axes = plt.subplots(1, 4, figsize=(S.DOUBLE, 52 * S.MM), gridspec_kw=dict(wspace=0.6))
    mb = pd.read_csv(RES / "mirror_benchmarks.csv")
    mb = mb[mb.connectome == "intact"]
    # a: JO benchmarks versus stimulation rate
    ax = axes[0]
    for side, col in [("left", S.LEFT), ("right", S.RIGHT)]:
        contra = "right" if side == "left" else "left"
        d = mb[(mb.cond == "jo") & (mb.side == side)].sort_values("rate")
        ax.plot(d.rate, d[f"aBN1_{side}"], "-o", color=col, ms=2.4, label=f"{side_label(side)}: aBN1 (ipsi)")
        ax.plot(d.rate, d[f"aDN1_{contra}"], "--s", color=col, ms=2.4, mfc="white", label=f"{side_label(side)}: aDN1 (contra)")
    ax.set_xlabel("JO stimulation (Hz)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_ylim(0, 150)
    ax.legend(loc="upper left", fontsize=4.5)
    S.panel_label(ax, "a", x=-0.38)
    # b: original neuron sets of Shiu et al.
    ax = axes[1]
    sh = pd.read_csv(RES / "shiu_sets.csv")
    sh = sh[sh.connectome == "intact"]
    for side, col, lab in [("left", S.LEFT, "Shiu set, left (20)"), ("right", S.RIGHT, "Shiu set, right (9)")]:
        d = sh[sh.side == side].sort_values("rate")
        ax.plot(d.rate, d.MN9, "-o", color=col, ms=2.4, label=lab)
    for side, col in [("left", S.LEFT), ("right", S.RIGHT)]:
        d = mb[(mb.cond == "sugar") & (mb.side == side) & (mb.rate >= 50)].sort_values("rate")
        ax.plot(d.rate, d[["MN9_left", "MN9_right"]].mean(axis=1), ":", color=col, lw=0.8)
    ax.set_xlabel("Sugar GRN stimulation (Hz)")
    ax.set_ylabel("MN9 firing rate (Hz)")
    ax.set_ylim(0, 125)
    ax.legend(loc="upper left", fontsize=4.8)
    ax.text(0.03, 0.74, "dotted: type-defined sets", transform=ax.transAxes, ha="left", fontsize=4.8, color=S.INK2)
    S.panel_label(ax, "b", x=-0.38)
    # c: leave-one-out over the 24 right sugar GRNs
    ax = axes[2]
    lo = pd.read_csv(RES / "right_sugar_leave_one_out.csv")
    lo = lo[lo.cond == "leave-one-out"].reset_index(drop=True)
    gate = lo.n_active.idxmin()
    ax.scatter(np.arange(len(lo)), lo.n_active, s=6, color=[S.AQUA if i == gate else S.RIGHT for i in range(len(lo))], lw=0)
    ax.set_yscale("log")
    ax.set_ylim(20, 20000)
    ax.axhline(2000, color=S.RED, lw=0.4, ls=":")
    ax.set_xlabel("Omitted right sugar GRN")
    ax.set_ylabel("Active neurons")
    ax.annotate(f"LB3b {lo.root_id[gate]}", (gate, lo.n_active[gate]), xytext=(1, 120), fontsize=4.8, color=S.AQUA)
    S.panel_label(ax, "c", x=-0.38)
    # d: synapses from each LB3b neuron onto antennal-lobe local neurons
    ax = axes[3]
    rng = np.random.default_rng(3)
    for i, side in enumerate(["left", "right"]):
        idx = [j for j in pops.stim("sugar", side) if a.cell_type.iloc[j] == "LB3b"]
        vals = []
        for j in idx:
            m = con.pre == j
            vals.append(con.count[m][(a.cell_class.to_numpy()[con.post[m]] == "ALLN")].sum())
        vals = np.array(vals)
        cols = [S.AQUA if (side == "right" and con.ids[j] == lo.root_id[gate]) else (S.LEFT if side == "left" else S.RIGHT) for j in idx]
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, s=8, color=cols, lw=0)
        ax.plot([i - 0.25, i + 0.25], [np.median(vals)] * 2, color=S.INK, lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Left LB3b", "Right LB3b"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("Synapses onto AL local neurons")
    S.panel_label(ax, "d", x=-0.38)
    save(fig, "ed4_mirror_controls")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["fig1", "fig2"]:
        globals()[name]()
