import matplotlib.pyplot as plt
import numpy as np
from src.utils_sparse_diag import binder_cumulant_computation

h_max = np.linspace(2 * np.exp(1) - 2, 2 * np.exp(1) + 5, 40)
n_dataset = 3000
ls = [5, 6, 7, 8, 9, 10]
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
            f"data/check_the_2nn_phase_transition/201222/binder_cumulant_range_h_{(h_max[0]):.2f}-{(h_max[-1]):.2f}_range_l_{ls[0]}-{ls[-1]}",
            hmax=h_max,
            ls=ls,
            u=u,
        )
