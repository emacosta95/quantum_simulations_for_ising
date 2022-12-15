import matplotlib.pyplot as plt
import numpy as np
from src.utils_sparse_diag import binder_cumulant_computation

h_max = [1.0, 2.0, 4.0, 5.44, 6.0, 7.0, 8.0]
n_dataset = 100
ls = [6, 8, 10, 12, 14, 16]
j1 = -1
j2 = -1
u = {}
for l in ls:
    for h in h_max:
        hs = np.random.uniform(0, h, size=(n_dataset, l))
        _, hs, us, es = binder_cumulant_computation(
            h_max=h,
            hs=hs,
            n_dataset=n_dataset,
            l=l,
            j1=j1,
            j2=j1,
            pbc=True,
            z_2=False,
            file_name="None",
            check_2nn=True,
            eps_breaking=0,
        )
        u[(l, h)] = us
        np.savez(
            f"data/check_the_2nn_phase_transition/151222/binder_cumulant_range_h_{len(h_max)}_range_l_{ls[0]}-{ls[-1]}",
            hmax=h_max,
            ls=ls,
            u=u,
        )
