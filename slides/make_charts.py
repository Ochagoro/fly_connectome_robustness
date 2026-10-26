"""Slide-sized charts for slides.html (1280 x 720 px slides).

Run: .venv/bin/python slides/make_charts.py
Reads results/*.csv, results/spikes/*.npz and data/ (read-only); writes slides/img/*.png only.
Figure sizes are given in slide pixels; text sizes in slide pixels (>= 16 px).
Colours follow the Digital Agency design-system tokens:
left = Blue-800, right = Orange-800, corrected = Green-700, neutral = Solid Gray 500.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "results", ROOT / "slides" / "img"
OUT.mkdir(exist_ok=True)

LEFT, RIGHT, FIXED = "#0031D8", "#C74700", "#1D8B56"
NEUTRAL, INK, INK2, LIGHT, GRID = "#7F7F7F", "#1A1A1A", "#4D4D4D", "#B3B3B3", "#CCCCCC"
PX = 0.72  # points per slide pixel at 100 dpi


def px(n):
    return n * PX


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
    "font.size": px(18),
    "axes.labelsize": px(18),
    "xtick.labelsize": px(17),
    "ytick.labelsize": px(17),
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.unicode_minus": True,
    "legend.frameon": False,
    "savefig.dpi": 200,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "lines.solid_capstyle": "round",
})


def fig_px(w, h):
    return plt.figure(figsize=(w / 100, h / 100), dpi=100)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("saved", name)


def thousands(ax, axis="y"):
    f = matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def mn9(df):
    return df[["MN9_left", "MN9_right"]].mean(axis=1)


# ------------------------------------------------------------------ data
mb = pd.read_csv(RES / "mirror_benchmarks.csv")
mb = mb[mb.connectome == "intact"]
rs = pd.read_csv(RES / "rescue.csv")


def cumulative_active(name, trials=3):
    curves = []
    t = np.arange(0, 10001, 10)
    for tr in range(trials):
        d = np.load(RES / f"spikes/{name}_trial{tr}.npz")
        stim = set(d["stim"].tolist())
        first = pd.Series(d["step"]).groupby(d["idx"]).min()
        first = first[~first.index.isin(stim)]
        curves.append(np.searchsorted(np.sort(first.to_numpy()), t, side="right"))
    return t * 0.1, np.array(curves)


# ------------------------------------------------------------------ 1. MN9 dose response
def mn9_curve():
    fig = fig_px(540, 420)
    ax = fig.add_axes([0.16, 0.17, 0.62, 0.76])
    for side, col, lab in [("left", LEFT, "左から刺激"), ("right", RIGHT, "右から刺激")]:
        d = mb[(mb.cond == "sugar") & (mb.side == side)].sort_values("rate")
        y = mn9(d)
        ax.plot(d.rate, y, "-o", color=col, lw=2.5, ms=6)
        ax.text(d.rate.iloc[-1] + 10, y.iloc[-1], lab, color=col, va="center", fontsize=px(18), fontweight="bold")
    ax.axvline(100, color=GRID, lw=1.2, ls=(0, (4, 3)), zorder=0)
    ax.text(104, 88, "ベンチマーク\n条件 100 Hz", color=INK2, fontsize=px(16), va="top", linespacing=1.3)
    ax.set_xlim(0, 310)
    ax.set_ylim(0, 92)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xlabel("砂糖 GRN の刺激頻度 (Hz)")
    ax.set_ylabel("MN9 の発火率 (Hz)")
    save(fig, "mn9_curve")


# ------------------------------------------------------------------ 2. active neurons vs rate (appendix)
def active_vs_rate():
    fig = fig_px(540, 400)
    ax = fig.add_axes([0.19, 0.18, 0.6, 0.75])
    for side, col, lab in [("left", LEFT, "左"), ("right", RIGHT, "右")]:
        d = mb[(mb.cond == "sugar") & (mb.side == side)].sort_values("rate")
        ax.plot(d.rate, d.n_active, "-o", color=col, lw=2.5, ms=6)
        ax.text(d.rate.iloc[-1] + 10, d.n_active.iloc[-1], lab, color=col, va="center", fontweight="bold")
    ax.axhline(2000, color=NEUTRAL, lw=1.2, ls=(0, (4, 3)))
    ax.text(305, 2300, "暴走の基準\n2,000 個", color=INK2, fontsize=px(16), ha="right", va="bottom", linespacing=1.3)
    ax.set_yscale("log")
    ax.set_ylim(20, 20000)
    ax.set_xlim(0, 310)
    ax.set_xticks([0, 100, 200, 300])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("砂糖 GRN の刺激頻度 (Hz)")
    ax.set_ylabel("発火したニューロン数")
    save(fig, "active_vs_rate")


# ------------------------------------------------------------------ 3. recruitment over time (runaway)
def recruit_time():
    fig = fig_px(580, 360)
    ax = fig.add_axes([0.16, 0.19, 0.62, 0.76])
    for name, col, lab in [("sugar_left_intact", LEFT, "左から刺激"), ("sugar_right_intact", RIGHT, "右から刺激")]:
        t, c = cumulative_active(name)
        for k in range(c.shape[0]):
            ax.plot(t, np.maximum(c[k], 0.8), color=col, lw=2.0, alpha=0.9)
        ax.text(612, c[:, t <= 600][:, -1].mean(), lab, color=col, va="center", fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlim(0, 600)
    ax.set_ylim(1, 20000)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("刺激開始からの時間 (ms)")
    ax.set_ylabel("発火したニューロン数（累積）")
    save(fig, "recruit_time")


# ------------------------------------------------------------------ 4. synapse ratio per population
def syn_ratio():
    # values: notes/results_summary.md (原因 1)
    cats = [("砂糖 GRN", 0.47), ("苦味 GRN", 0.64), ("Ir94e GRN", 0.73), ("水 GRN", 0.36), ("JO（触角）", 1.18)]
    fig = fig_px(520, 400)
    ax = fig.add_axes([0.27, 0.2, 0.66, 0.74])
    y = np.arange(len(cats))[::-1]
    for yi, (lab, v) in zip(y, cats):
        col = RIGHT if lab.startswith("砂糖") else NEUTRAL
        ax.barh(yi, v, height=0.62, color=col)
        ax.text(v + 0.03, yi, f"{v:.2f}", va="center", color=col if col == RIGHT else INK, fontweight="bold")
    ax.axvline(1, color=INK, lw=1.2, ls=(0, (4, 3)))
    ax.text(1.03, len(cats) - 0.35, "左右同数", ha="left", va="bottom", color=INK2, fontsize=px(16))
    ax.set_yticks(y)
    ax.set_yticklabels([c[0] for c in cats])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1.4)
    ax.set_ylim(-0.6, len(cats) - 0.1)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("右 ÷ 左（1 個あたりの出力シナプス数）")
    save(fig, "syn_ratio")


# ------------------------------------------------------------------ connectome-derived panels
def load_ctx():
    sys.path.insert(0, str(ROOT / "src"))
    from neurons import Populations  # noqa: E402
    from sim import load_connectome  # noqa: E402
    con = load_connectome("783")
    return con, Populations(con)


def deletion(con, pops):
    out_syn = np.bincount(con.pre, weights=con.count, minlength=con.n)
    totL = out_syn[pops.stim("sugar", "left")].sum()
    totR = out_syn[pops.stim("sugar", "right")].sum()
    print("sugar GRN output synapses L/R", totL, totR)
    td = pd.read_csv(RES / "targeted_sugar_deletion.csv")
    td["MN9"] = mn9(td)
    d = td[(td.connectome == "al_fix") & (td.rate == 100)].groupby("frac_deleted").MN9.agg(["mean", "min", "max"])
    x = (1 - d.index.to_numpy()) * totL / 1000
    rr = rs[(rs.variant == "al_fix") & (rs.cond == "sugar") & (rs.side == "right")]
    right_mn9 = mn9(rr).mean()
    fig = fig_px(580, 420)
    ax = fig.add_axes([0.14, 0.17, 0.82, 0.78])
    ax.fill_between(x, d["min"], d["max"], color=LEFT, alpha=0.15, lw=0)
    ax.plot(x, d["mean"], "-o", color=LEFT, lw=2.5, ms=6)
    ax.text(x[0] - 0.1, d["mean"].iloc[0] + 5, "左の砂糖 GRN から\nシナプスを抜いていく", color=LEFT, ha="right",
            va="bottom", fontweight="bold", linespacing=1.35)
    ax.plot([totR / 1000], [right_mn9], "D", color=RIGHT, ms=11, mec="white", mew=1.5, zorder=5)
    ax.annotate("右の実測\n（シナプス 3,000 個）", xy=(totR / 1000, right_mn9), xytext=(1.9, 20),
                color=RIGHT, fontweight="bold", ha="center", va="bottom", linespacing=1.35,
                arrowprops=dict(arrowstyle="-", lw=1.3, color=RIGHT, shrinkB=7))
    ax.axvline(5, color=GRID, lw=1.2, ls=(0, (4, 3)), zorder=0)
    ax.text(5.08, 1.5, "5,000 個", color=INK2, fontsize=px(16))
    ax.set_xlim(0, 9.4)
    ax.set_ylim(0, 75)
    ax.set_xticks([0, 2, 4, 6, 8])
    ax.set_xlabel("刺激した砂糖 GRN の出力シナプス数（×1,000）")
    ax.set_ylabel("MN9 の発火率 (Hz)")
    save(fig, "deletion")


def abn1(con, pops):
    in_syn = np.bincount(con.post, weights=con.count, minlength=con.n)
    L = in_syn[pops.readout("aBN1", "left")].sum()
    R = in_syn[pops.readout("aBN1", "right")].sum()
    print("aBN1 inputs L/R", L, R)
    # direct JO -> aBN1 (summary: left 103, right 35)
    vals = [("入力シナプスの総数", 1833, 459), ("JO からの直接入力", 103, 35)]
    fig = fig_px(520, 400)
    axs = [fig.add_axes([0.2, 0.12, 0.3, 0.72]), fig.add_axes([0.68, 0.12, 0.3, 0.72])]
    for ax, (title, l, r) in zip(axs, vals):
        ax.bar([0, 1], [l, r], width=0.62, color=[LEFT, RIGHT])
        for xi, v, c in [(0, l, LEFT), (1, r, RIGHT)]:
            ax.text(xi, v * 1.03, f"{v:,}", ha="center", va="bottom", color=c, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["左", "右"])
        for t, c in zip(ax.get_xticklabels(), [LEFT, RIGHT]):
            t.set_color(c)
            t.set_fontweight("bold")
        ax.set_ylim(0, max(l, r) * 1.2)
        ax.set_xlim(-0.6, 1.6)
        thousands(ax)
        ax.set_title(title, fontsize=px(18), pad=18, loc="left", x=-0.35)
    axs[0].set_ylabel("シナプス数")
    save(fig, "abn1")


def nt_conf(pops):
    a = pops.ann
    ln = a[a.cell_class == "ALLN"]
    groups = [("セロトニン", "serotonin"), ("GABA", "gaba"), ("グルタミン酸", "glutamate"), ("アセチル\nコリン", "acetylcholine")]
    # median labels follow notes/results_summary.md (serotonin 0.35, GABA 0.52)
    med_label = {"serotonin": "0.35", "gaba": "0.52"}
    rng = np.random.default_rng(0)
    fig = fig_px(560, 340)
    ax = fig.add_axes([0.15, 0.22, 0.82, 0.74])
    for i, (lab, key) in enumerate(groups):
        v = ln[ln.top_nt == key].top_nt_conf.dropna().to_numpy()
        col = RIGHT if key == "serotonin" else LIGHT
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=22, lw=0, alpha=0.85 if key == "serotonin" else 0.7, color=col)
        med = np.median(v)
        ax.plot([i - 0.3, i + 0.3], [med] * 2, color=INK, lw=2.5)
        print("ALLN", key, len(v), round(med, 3))
        if key in ("serotonin", "gaba"):
            ax.text(i + 0.33, med, med_label[key], va="center", fontweight="bold", color=INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels([g[0] for g in groups])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 3.6)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("伝達物質の予測の確信度")
    for t in ax.get_xticklabels():
        t.set_linespacing(1.2)
    save(fig, "nt_conf")


def alfix_recruit():
    fig = fig_px(560, 420)
    ax = fig.add_axes([0.17, 0.17, 0.6, 0.78])
    for name, col, lab in [("sugar_right_intact", RIGHT, "右・公開版"),
                           ("sugar_left_intact", LEFT, "左・公開版"),
                           ("sugar_right_alfix", FIXED, "右・40 個を\n抑制性に補正")]:
        t, c = cumulative_active(name)
        m = c.mean(0)
        ax.plot(t, np.maximum(m, 0.8), color=col, lw=2.5)
        ax.text(1020, m[-1], lab, color=col, va="center", fontweight="bold", linespacing=1.3)
    ax.set_yscale("log")
    ax.set_xlim(0, 1000)
    ax.set_ylim(1, 20000)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("刺激開始からの時間 (ms)")
    ax.set_ylabel("発火したニューロン数（累積）")
    save(fig, "alfix_recruit")


# ------------------------------------------------------------------ screen (Fig. 4, partial numbers)
def screen():
    # runaway types (either side), results/screen_summary.csv; matches notes/results_summary.md (図 4)
    S = pd.read_csv(RES / "screen_summary.csv").rename(columns={"Unnamed: 0": "variant"})
    cnt = {v: int((S[S.variant == v].run_L | S[S.variant == v].run_R).sum()) for v in ["intact", "al_fix", "corrected"]}
    assert (cnt["intact"], cnt["al_fix"], cnt["corrected"]) == (60, 8, 1), cnt
    fig = fig_px(560, 290)
    ax = fig.add_axes([0.36, 0.2, 0.58, 0.76])
    rows = [("公開版の配線図", cnt["intact"], NEUTRAL, "60 / 131（46%）"),
            ("＋触角葉の 40 個を補正", cnt["al_fix"], FIXED, "8 / 131（6%）"),
            ("＋細胞タイプ単位で\n符号をそろえる", cnt["corrected"], FIXED, "1 / 131")]
    for yi, (lab, n, col, txt) in zip([2, 1, 0], rows):
        ax.barh(yi, n / 131 * 100, height=0.6, color=col)
        ax.text(n / 131 * 100 + 1.5, yi, txt, va="center", fontweight="bold", color=INK)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([r[0] for r in rows], linespacing=1.25)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel("暴走を起こした感覚細胞タイプ (%)")
    save(fig, "screen")


def mirror_r():
    S = pd.read_csv(RES / "screen_summary.csv").rename(columns={"Unnamed: 0": "variant"})
    a = S[S.variant == "al_fix"]
    a = a[~(a.run_L | a.run_R)].dropna(subset=["r_LR", "r_rep"])  # runaway types excluded
    assert len(a) == 118
    rng = np.random.default_rng(1)
    fig = fig_px(520, 330)
    ax = fig.add_axes([0.17, 0.21, 0.62, 0.75])
    for i, (col_name, col, lab, med_txt) in enumerate([("r_rep", NEUTRAL, "同じ側で\n2 回", "0.99"),
                                                      ("r_LR", INK, "左と右", "0.58")]):
        v = a[col_name].to_numpy()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=18, lw=0, alpha=0.6, color=col)
        med = np.median(v)
        ax.plot([i - 0.28, i + 0.28], [med] * 2, color=INK, lw=2.5)
        ax.text(i + 0.32, med, med_txt, va="center", fontweight="bold")
    lo = a.r_rep.min()
    ax.axhline(lo, color=NEUTRAL, lw=1.2, ls=(0, (4, 3)), zorder=0)
    ax.text(1.62, lo, "繰り返しの\n最低値", va="center", ha="left", color=INK2, fontsize=px(16), linespacing=1.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["同じ側で\n2 回", "左と右"], linespacing=1.25)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylim(-0.2, 1.05)
    ax.set_ylabel("脳全体の応答の相関 $r$")
    save(fig, "mirror_r")


def error_dose():
    e = pd.read_csv(RES / "error_metrics.csv")
    g = e.groupby(["model", "level"])[["sugar_rel", "aDN1_rel", "runaway"]].mean()
    base = g.loc["intact"].iloc[0]

    def series(m, col):
        d = g.loc[m][col]
        return np.r_[0, d.index.to_numpy()], np.r_[base[col], d.to_numpy()]

    fig = fig_px(640, 420)
    # a: missing vs false synapses
    ax = fig.add_axes([0.12, 0.2, 0.36, 0.72])
    x, y = series("deletion", "sugar_rel")
    ax.plot(x * 100, y, "-o", color=INK, lw=2.5, ms=6)
    x, y = series("deletion", "aDN1_rel")
    ax.plot(x * 100, y, "--o", color=INK, lw=2.0, ms=6, mfc="white")
    ax.text(90, 0.07, "見逃し（削除）", color=INK, fontweight="bold", ha="right", va="bottom")
    ax.text(24, 0.56, "砂糖 → MN9", color=INK, ha="left", va="center")
    ax.text(31, 0.27, "JO → aDN1", color=INK, ha="left", va="center")
    x, y = series("false_pos", "sugar_rel")
    ax.plot(x * 100, y, "-s", color=NEUTRAL, lw=2.5, ms=6)
    ax.text(80, 1.08, "偽のシナプス（追加）", color=INK2, fontweight="bold", ha="right", va="bottom")
    ax.axhline(0.5, color=GRID, lw=1.2, zorder=0)
    ax.set_xlim(0, 92)
    ax.set_ylim(-0.05, 1.3)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("脳全体で変えたシナプス (%)")
    ax.set_ylabel("応答（公開版 = 1）")
    # b: random sign errors
    ax = fig.add_axes([0.62, 0.2, 0.36, 0.72])
    x, y = series("sign_flip", "sugar_rel")
    ax.plot(x * 100, y, "-o", color=INK, lw=2.5, ms=6)
    ax.text(10.8, 0.64, "砂糖 → MN9", color=INK, fontweight="bold")
    x, yb = series("sign_flip", "aDN1_rel")
    ax.plot(x * 100, yb, "--o", color=INK, lw=2.0, ms=6, mfc="white")
    ax.text(9.5, 0.2, "JO → aDN1", color=INK, fontweight="bold", ha="right")
    ax.annotate("暴走\n3 回中 2 回", xy=(20, 0.42), xytext=(15.5, 1.0), ha="center", va="bottom", color="#EC0000",
                fontsize=px(16), fontweight="bold", linespacing=1.25,
                arrowprops=dict(arrowstyle="-", lw=1.2, color="#EC0000", shrinkB=6))
    ax.axhline(0.5, color=GRID, lw=1.2, zorder=0)
    ax.set_xlim(0, 21)
    ax.set_ylim(-0.05, 1.3)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("符号を逆にしたニューロン (%)")
    save(fig, "error_dose")


# ------------------------------------------------------------------ validation (appendix)
def leave_one_out():
    d = pd.read_csv(RES / "right_sugar_leave_one_out.csv")
    d = d[d.cond == "leave-one-out"].reset_index(drop=True)
    assert len(d) == 24
    key = d.root_id == 720575940629884119
    assert int(d[key].n_active.iloc[0]) == 58
    fig = fig_px(560, 340)
    ax = fig.add_axes([0.17, 0.2, 0.78, 0.75])
    xs = np.arange(1, 25)
    ax.scatter(xs[~key], d.n_active[~key], s=60, color=RIGHT, lw=0, zorder=3)
    ax.scatter(xs[key], d.n_active[key], s=110, marker="D", color=INK, lw=0, zorder=3)
    k = int(xs[key][0])
    ax.annotate("LB3b の 1 個を除くと 58 個", xy=(k, 58), xytext=(k - 0.8, 200), ha="right", va="bottom",
                color=INK, fontweight="bold", arrowprops=dict(arrowstyle="-", lw=1.2, color=INK, shrinkB=7))
    ax.text(0.8, 11000, "ほかの 23 個のどれを除いても暴走が続く", color=RIGHT, fontweight="bold", va="bottom")
    ax.axhline(2000, color=NEUTRAL, lw=1.2, ls=(0, (4, 3)))
    ax.text(24.3, 2300, "暴走の基準", color=INK2, fontsize=px(16), ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(20, 40000)
    ax.set_xlim(0, 25)
    ax.set_xticks([1, 6, 12, 18, 24])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_xlabel("除いた右の砂糖 GRN（24 個のうち 1 個）")
    ax.set_ylabel("発火したニューロン数")
    save(fig, "leave_one_out")


def shiu_sets():
    d = pd.read_csv(RES / "shiu_sets.csv")
    d = d[d.connectome == "intact"]
    fig = fig_px(600, 420)
    ax = fig.add_axes([0.13, 0.17, 0.55, 0.78])
    for side, col in [("left", LEFT), ("right", RIGHT)]:
        t = mb[(mb.cond == "sugar") & (mb.side == side)].sort_values("rate")
        ax.plot(t.rate, mn9(t), ":", color=col, lw=2.2)
        g = d[d.side == side].sort_values("rate")
        ax.plot(g.rate, g.MN9, "-o", color=col, lw=2.5, ms=6)
    ax.text(310, 88, "左・Shiu らの組\n（20 個）", color=LEFT, va="center", fontweight="bold", linespacing=1.25)
    ax.text(310, 71, "左・細胞タイプ全体\n（33 個、点線）", color=LEFT, va="center", linespacing=1.25)
    ax.text(310, 52.4, "右・Shiu らの組\n（9 個）", color=RIGHT, va="center", fontweight="bold", linespacing=1.25)
    ax.text(310, 16.5, "右・細胞タイプ全体\n（24 個、点線）", color=RIGHT, va="center", linespacing=1.25)
    ax.set_xlim(0, 305)
    ax.set_ylim(0, 92)
    ax.set_xticks([0, 100, 200, 300])
    ax.set_xlabel("砂糖 GRN の刺激頻度 (Hz)")
    ax.set_ylabel("MN9 の発火率 (Hz)")
    save(fig, "shiu_sets")


def validation():
    items = [("validation_sugarL_783_150Hz", "左・砂糖 150 Hz", LEFT),
             ("validation_sugar_left_783_100Hz", "左・砂糖 100 Hz", LEFT),
             ("validation_sugar_right_783_100Hz", "右・砂糖 100 Hz", RIGHT)]
    fig = fig_px(1150, 340)
    W = 0.185
    for k, (f, title, col) in enumerate(items):
        ax = fig.add_axes([0.06 + k * 0.245, 0.21, W, 0.64])
        x = pd.read_csv(RES / f"{f}.csv", index_col=0)
        ax.scatter(x.brian2, x.gpu, s=10, color=col, lw=0, alpha=0.5, rasterized=True)
        m = max(x.brian2.max(), x.gpu.max()) * 1.05
        ax.plot([0, m], [0, m], color=NEUTRAL, lw=1, ls=(0, (4, 3)))
        r = np.corrcoef(x.brian2, x.gpu)[0, 1]
        ax.text(0.05, 0.97, f"$r$ = {r:.4f}\n$n$ = {len(x):,}", transform=ax.transAxes, va="top", linespacing=1.4)
        ax.set_xlim(0, m)
        ax.set_ylim(0, m)
        ax.set_title(title, fontsize=px(18), loc="left", pad=10)
        ax.set_xlabel("Brian 2 (Hz)")
        if k == 0:
            ax.set_ylabel("GPU 実装 (Hz)")
    ax = fig.add_axes([0.06 + 3 * 0.245, 0.21, W, 0.64])
    v = pd.read_parquet(RES / "version_rates.parquet").dropna()
    a, b = v["v630_100Hz"], v["v783_100Hz"]
    mm = (a > 0) | (b > 0)
    ax.scatter(a[mm], b[mm], s=10, color=INK2, lw=0, alpha=0.5, rasterized=True)
    ax.plot([0, 200], [0, 200], color=NEUTRAL, lw=1, ls=(0, (4, 3)))
    ax.text(0.05, 0.97, f"$r$ = {np.corrcoef(a[mm], b[mm])[0, 1]:.3f}\n$n$ = {mm.sum():,}", transform=ax.transAxes,
            va="top", linespacing=1.4)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_title("版の比較・左 100 Hz", fontsize=px(18), loc="left", pad=10)
    ax.set_xlabel("v630 (Hz)")
    ax.set_ylabel("v783 (Hz)")
    save(fig, "validation")


if __name__ == "__main__":
    mn9_curve()
    active_vs_rate()
    recruit_time()
    syn_ratio()
    alfix_recruit()
    screen()
    mirror_r()
    error_dose()
    leave_one_out()
    shiu_sets()
    validation()
    con, pops = load_ctx()
    deletion(con, pops)
    abn1(con, pops)
    nt_conf(pops)
