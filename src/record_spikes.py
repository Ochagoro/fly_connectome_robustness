"""Record every spike of single 1 s trials for the cascade figure and the video."""
import numpy as np
from sim import load_connectome, Simulator, ROOT
from neurons import Populations
from experiments import serotonin_fixed

con = load_connectome("783"); pops = Populations(con)
fixed = serotonin_fixed(con, pops.ann)
runs = [("sugar_left_intact", con, "left", 100.0), ("sugar_right_intact", con, "right", 100.0),
        ("sugar_right_5ht", fixed, "right", 100.0), ("sugar_right_5ht_300Hz", fixed, "right", 300.0),
        ("sugar_left_300Hz", con, "left", 300.0)]
for name, c, side, hz in runs:
    sim = Simulator(c)
    for trial in range(3):
        out = sim.run([[(pops.stim("sugar", side), hz)]], seed=50 + trial, record_all=True)
        steps = np.concatenate([np.full(len(f), t, np.int32) for t, f in out["events"]])
        idx = np.concatenate([f.numpy() % con.n for t, f in out["events"]]).astype(np.int32)
        np.savez_compressed(ROOT / f"results/spikes/{name}_trial{trial}.npz", step=steps, idx=idx,
                            stim=pops.stim("sugar", side))
        print(name, trial, "spikes", len(idx), "active", len(np.unique(idx)), flush=True)
