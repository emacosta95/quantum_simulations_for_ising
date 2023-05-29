# %% Imports
import numpy as np
import quspin
from src.utils_sparse_diag import (
    adj_generator,
    ising_coupling,
    ising_external_field,
    compute_magnetization,
    functional_f,
    get_gs,
)
from tqdm import trange

# %% Seed and Data
l = 8
ndata = 6000
basis = quspin.basis.spin_basis_1d(l)
h_max = np.e

h = np.random.uniform(0, h_max, size=(ndata, l))
z = np.zeros_like(h)
f = np.zeros(shape=(ndata, 2, l))
e = np.zeros(shape=(ndata))
x = np.zeros(shape=(ndata, l))
# scheme of the hamiltonian: ZZ X Z
# define the adj in the ising model x_i x_i+1 with PBC
ad_j = {}
for i in range(l):
    ad_j[(i, (i + 1) % l)] = 1.0

ham_zz, f_op = ising_coupling(adj=ad_j, l=l, basis=basis, direction="zz")
ham_x = ising_external_field(h=np.ones(l), l=l, basis=basis, direction="x")

for i in trange(ndata):
    ham_z = ising_external_field(h=h[i], l=l, basis=basis, direction="z")

    ham = ham_x + ham_zz + ham_z
    e_i, psi = get_gs(ham=ham, eightype="Standard", basis=basis, k=1)

    f_i = functional_f(psi=psi, l=l, f_density_op=f_op)
    x_i = compute_magnetization(psi=psi, l=l, basis=basis, direction="x")
    z_i = compute_magnetization(psi=psi, l=l, basis=basis, direction="z")

    e[i] = e_i
    z[i] = z_i
    f[i, 0] = f_i.mean(-1)
    f[i, 1] = x_i
    x[i] = x_i

    if i % 1000 == 0:
        np.savez(
            f"data/kohm_sham_approach/dataset_2channels_h_{h_max:.1f}_j_1_1nn_n_{ndata}",
            energy=e,
            potential=h,
            density=z,
            density_F=f,
            transverse_magnetization=x,
        )

np.savez(
    f"data/kohm_sham_approach/dataset_2channels_h_{h_max:.1f}_j_1_1nn_n_{ndata}",
    energy=e,
    potential=h,
    density=z,
    density_F=f,
    transverse_magnetization=x,
)

# %%
