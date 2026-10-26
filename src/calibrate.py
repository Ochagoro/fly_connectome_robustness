"""Intact responses of every benchmark from each hemisphere at several stimulus rates."""
import time, numpy as np, pandas as pd
from sim import load_connectome, Simulator, ROOT
from neurons import Populations, SIDES, READOUT_TYPES, STIM_TYPES

con = load_connectome("783"); pops = Populations(con); sim = Simulator(con)
for k in list(STIM_TYPES):
    print(k, {s: len(pops.stim(k, s)) for s in SIDES})
for k in READOUT_TYPES:
    print(k, {s: len(pops.readout(k, s)) for s in SIDES})
NT = 8
rows = []
for rate in [50., 100., 150., 200.]:
    conds = []
    for s in SIDES:
        conds += [("sugar", s, [(pops.stim("sugar", s), rate)]),
                  ("water", s, [(pops.stim("water", s), rate)]),
                  ("sugar+bitter", s, [(pops.stim("sugar", s), rate), (pops.stim("bitter", s), rate)]),
                  ("sugar+ir94e", s, [(pops.stim("sugar", s), rate), (pops.stim("ir94e", s), rate)]),
                  ("jo", s, [(pops.stim("jo", s), rate)])]
    stim = [c[2] for c in conds for _ in range(NT)]
    t0 = time.time(); out = sim.run(stim, seed=int(rate)); dt = time.time() - t0
    r = out["counts"].reshape(len(conds), NT, -1).mean(1)
    for ci, (name, s, _) in enumerate(conds):
        row = dict(rate=rate, cond=name, side=s, n_active=int((r[ci] > 1).sum()))
        for k in READOUT_TYPES:
            for rs in SIDES:
                row[f"{k}_{rs}"] = r[ci, pops.readout(k, rs)].mean()
        rows.append(row)
    print("rate", rate, "batch of", len(stim), "took %.1f s" % dt)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(df.round(1).to_string())
df.to_csv(ROOT / "results/calibration_intact.csv", index=False)
