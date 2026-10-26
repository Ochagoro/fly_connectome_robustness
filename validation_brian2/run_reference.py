"""Run the original Brian 2 model (Shiu et al.) on FlyWire v783 for the reference comparison."""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data/shiu_model"))
from model import run_exp, default_params as params
from brian2 import Hz
import pandas as pd

D = Path(__file__).resolve().parents[1] / "data"
ann = pd.read_csv(D / "neuron_annotations.tsv", sep="\t", usecols=["root_id", "cell_type", "side"])
comp = pd.read_csv(D / "shiu_model/Completeness_783.csv", index_col=0)
ann = ann[ann.root_id.isin(comp.index)]
sugar = ann[(ann.cell_type.isin(["LB3b", "LB3c"])) & (ann.side == "left")].root_id.tolist()
json.dump(sugar, open("stim_sugar_left_783.json", "w"))
params["n_run"] = 30
params["r_poi"] = 150 * Hz
t0 = time.time()
run_exp(exp_name="sugarL_783_150Hz", neu_exc=sugar, path_res="./out",
        path_comp=str(D / "shiu_model/Completeness_783.csv"),
        path_con=str(D / "shiu_model/Connectivity_783.parquet"),
        params=params, n_proc=15, force_overwrite=True)
print("done in %.0f s" % (time.time() - t0))
