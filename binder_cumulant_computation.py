import matplotlib.pyplot as plt
import numpy as np
from src.utils_sparse_diag import binder_cumulant_computation

h_max = np.linspace(2, 4, 200)  # np.linspace(2 * np.exp(1) - 2, 2 * np.exp(1) + 2, 200)
n_dataset = 3000
ls = [6, 8, 10, 12, 14]
j1 = 1
j2 = 1
u = {}
x2 = {}
x4 = {}
for l in ls:
    for h in h_max:
        hs = np.random.uniform(0, h, size=(n_dataset, l))
        _, hs, us, x2s, x4s = binder_cumulant_computation(
            h_max=h,
            hs=hs,
            n_dataset=n_dataset,
            l=l,
            j1=j1,
            j2=j1,
            pbc=True,
            z_2=False,
            file_name="None",
            check_2nn=False,
            eps_breaking=0.1,
        )
        u[(l, h)] = us
        x2[(l, h)] = x2s
        x4[(l, h)] = x4s

        np.savez(
            f"data/1nn_xx_z_x/110123/binder_cumulant_range_h_{(h_max[0]):.2f}-{(h_max[-1]):.2f}_range_l_{ls[0]}-{ls[-1]}",
            hmax=h_max,
            ls=ls,
            u=u,
            x2=x2,
            x4=x4,
        )
