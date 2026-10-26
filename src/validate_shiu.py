"""Check the GPU simulator against the published Brian 2 output (v630, sugar GRNs, 150 Hz, 30 trials)."""
import json, time
import numpy as np, pandas as pd
from sim import load_connectome, Simulator, ROOT, DATA

ids = json.load(open(DATA / "shiu_id_lists_v630.json"))["neu_sugar"]
con = load_connectome("630")
sim = Simulator(con)
idx = con.index_of(ids)
print("edges", sim.n_edges, "stim", len(idx))
t0 = time.time()
out = sim.run([[(idx, 150.0)]] * 30, seed=1)
print("30 trials took %.1f s" % (time.time() - t0))
rate = out["counts"].mean(0) / (out["t_run"] * 1e-3)
ref = pd.read_parquet(DATA / "shiu_model/results/example/sugarR.parquet")
ref_rate = ref.groupby("flywire_id").size() / 30.0
mine = pd.Series(rate, index=con.ids)
both = pd.concat([mine.rename("gpu"), ref_rate.rename("brian2")], axis=1).fillna(0)
both = both[(both.gpu > 0) | (both.brian2 > 0)]
both = both[~both.index.isin(ids)]
print("neurons active in either:", len(both))
print("pearson r = %.4f" % np.corrcoef(both.gpu, both.brian2)[0, 1])
print("MN9 (720575940660219265): gpu %.2f Hz, brian2 %.2f Hz" % (both.gpu.get(720575940660219265, 0), both.brian2.get(720575940660219265, 0)))
print(both.sort_values("brian2", ascending=False).head(15).to_string())
both.to_csv(ROOT / "results/validation_v630_sugar150.csv")
