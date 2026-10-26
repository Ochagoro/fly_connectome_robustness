"""Additional analyses added during review. Run: python analysis_extra.py [shiu_sets leave_one_out screen_recount lateral]

  shiu_sets      MN9 responses to the original neuron sets of Shiu et al. (left 'sugarR', right 'sugarL')
  leave_one_out  omit each right sugar GRN in turn to find the runaway entry neuron
  screen_recount count active neurons in the sensory screen without the stimulated neurons
  lateral        share of evoked activity in the stimulated hemisphere for every screen condition
"""
import json
import sys

import numpy as np
import pandas as pd

import perturb as P
from experiments import run_conditions
from neurons import Populations, load_annotations
from sim import DATA, ROOT, Simulator, load_connectome

RES = ROOT / "results"


def shiu_sets():
    con = load_connectome("783"); pops = Populations(con)
    L = json.load(open(DATA / "shiu_id_lists_v630.json"))
    sets = [("sugarR", "left", con.index_of(L["neu_sugar"])), ("sugarL", "right", con.index_of(L["neu_sugar_left"]))]
    rows = []
    for lab, c in [("intact", con), ("al_fix", P.al_serotonin_fix(con, pops.ann))]:
        conds = [dict(cond=f"Shiu set ({nm})", side=sd, rate=float(r), stim=[(ix, float(r))])
                 for nm, sd, ix in sets for r in [50, 100, 150, 200, 300]]
        df = run_conditions(Simulator(c), pops, conds, n_trials=8, seed=77)
        df["connectome"] = lab
        rows.append(df)
    df = pd.concat(rows)
    df["MN9"] = df[["MN9_left", "MN9_right"]].mean(axis=1)
    df.to_csv(RES / "shiu_sets.csv", index=False)


def leave_one_out():
    con = load_connectome("783"); pops = Populations(con); a = pops.ann
    R = pops.stim("sugar", "right")
    conds = [dict(cond="all", k=-1, rate=100.0, stim=[(R, 100.0)])]
    for t in ["LB3b", "LB3c"]:
        conds.append(dict(cond=f"only {t}", k=-1, rate=100.0, stim=[(R[a.iloc[R].cell_type.to_numpy() == t], 100.0)]))
    for k, j in enumerate(R):
        conds.append(dict(cond="leave-one-out", k=int(j), rate=100.0, stim=[(np.delete(R, k), 100.0)]))
    df = run_conditions(Simulator(con), pops, conds, n_trials=4, seed=31)
    df["MN9"] = df[["MN9_left", "MN9_right"]].mean(axis=1)
    df["root_id"] = [con.ids[k] if k >= 0 else 0 for k in df.k]
    df["cell_type"] = [a.cell_type.iloc[k] if k >= 0 else "" for k in df.k]
    df["top_nt"] = [a.top_nt.iloc[k] if k >= 0 else "" for k in df.k]
    out = np.bincount(con.pre, weights=con.count, minlength=con.n)
    df["out_syn"] = [out[k] if k >= 0 else 0 for k in df.k]
    df.to_csv(RES / "right_sugar_leave_one_out.csv", index=False)


def screen_recount():
    con = load_connectome("783")
    a = load_annotations().set_index("root_id").reindex(con.ids)
    ct, side = a.cell_type.to_numpy(), a.side.to_numpy()
    for v in ["intact", "al_fix", "corrected"]:
        d = pd.read_csv(RES / f"screen_{v}.csv")
        R = np.load(RES / f"screen/rates_{v}.npy")
        act = []
        for i, r in d.iterrows():
            x = R[i].astype(np.float32)
            x[(ct == r.cell_type) & (side == r.side)] = 0
            act.append(int((x > 0).sum()))
        d["n_active_nonstim"] = act
        d.to_csv(RES / f"screen_{v}.csv", index=False)


def lateral():
    con = load_connectome("783")
    a = load_annotations().set_index("root_id").reindex(con.ids)
    side = a.side.to_numpy()
    L, Rm = side == "left", side == "right"
    F = pd.read_csv(RES / "screen_features.csv").set_index("cell_type")
    for v in ["intact", "al_fix", "corrected"]:
        d = pd.read_csv(RES / f"screen_{v}.csv")
        rates = np.load(RES / f"screen/rates_{v}.npy").astype(np.float32)
        rows = []
        for i, r in d.iterrows():
            x = rates[i].copy()
            x[(a.cell_type.to_numpy() == r.cell_type) & (side == r.side)] = 0
            tot = x[L].sum() + x[Rm].sum()
            rows.append(dict(cell_type=r.cell_type, side=r.side, left_frac=x[L].sum() / tot if tot > 0 else np.nan,
                             n_active=r.n_active, cls=F.loc[r.cell_type, "cell_class"]))
        pd.DataFrame(rows).to_csv(RES / f"screen_lateral_{v}.csv", index=False)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        globals()[name]()
        print("done", name, flush=True)
