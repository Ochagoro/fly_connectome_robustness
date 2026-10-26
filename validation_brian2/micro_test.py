import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data/shiu_model"))
from model import default_params as p
from brian2 import NeuronGroup, Synapses, PoissonInput, SpikeMonitor, StateMonitor, Network, ms, mV, Hz, defaultclock, seed
import numpy as np
seed(1)
neu = NeuronGroup(3, model=p['eqs'], method='linear', threshold=p['eq_th'], reset=p['eq_rst'], refractory='rfc', namespace=p)
neu.v = p['v_0']; neu.g = 0; neu.rfc = p['t_rfc']
syn = Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=p['t_dly'])
syn.connect(i=[0, 0], j=[1, 2]); syn.w = [30 * p['w_syn'], 60 * p['w_syn']]
pi = PoissonInput(target=neu[0:1], target_var='v', N=1, rate=100*Hz, weight=p['w_syn']*p['f_poi'])
neu[0].rfc = 0*ms
sm = SpikeMonitor(neu)
st = StateMonitor(neu, ['v','g'], record=[1,2], dt=0.1*ms)
net = Network(neu, syn, pi, sm, st); net.run(20000*ms)
c = np.bincount(np.asarray(sm.i), minlength=3) / 20.0
print("brian2 rates", c)
t0 = np.asarray(sm.t[sm.i == 0] / ms)[:3]; print("first input spikes (ms)", t0)
k = int(round((t0[0] + 1.8) / 0.1))
print("v1 trace around first arrival", np.round(np.asarray(st.v[0][k-2:k+8] / mV), 4))
print("g1 trace", np.round(np.asarray(st.g[0][k-2:k+8] / mV), 4))
np.save("micro_input_spikes.npy", np.asarray(sm.t[sm.i == 0] / ms))
np.save("micro_n2_spikes.npy", np.asarray(sm.t[sm.i == 2] / ms))
st2 = StateMonitor
vv = np.asarray(st.v[1] / mV); gg = np.asarray(st.g[1] / mV)
np.save("micro_n2_v.npy", vv); np.save("micro_n2_g.npy", gg)
