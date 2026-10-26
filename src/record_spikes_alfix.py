"""Spike recordings for the right hemisphere with the antennal-lobe sign fix only."""
import numpy as np
from sim import load_connectome, Simulator, ROOT
from neurons import Populations
import perturb as P
con = load_connectome("783"); pops = Populations(con)
sim = Simulator(P.al_serotonin_fix(con, pops.ann))
for trial in range(3):
    out = sim.run([[(pops.stim("sugar", "right"), 100.0)]], seed=50 + trial, record_all=True)
    steps = np.concatenate([np.full(len(f), t, np.int32) for t, f in out["events"]])
    idx = np.concatenate([f.numpy() % con.n for t, f in out["events"]]).astype(np.int32)
    np.savez_compressed(ROOT / f"results/spikes/sugar_right_alfix_trial{trial}.npz", step=steps, idx=idx,
                        stim=pops.stim("sugar", "right"))
    print("alfix", trial, len(np.unique(idx)), flush=True)
