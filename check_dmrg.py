# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

from src.utils_exact_diagonalization import QuantumSpinSystem
from src.utils_sparse_diagonalization import transverse_ising_sparse_simulator_sample

l = 16
j1 = -1
eps_breaking = 10**-2
data = np.load(
    "data/dataset_dmrg/test_unet_periodic_1nn_l_16_h_2.71_ndata_10.npz")

f_dmrg = data["F"]
z_dmrg = data["density"]
dens_f_dmrg = data["density_F"]
pot = data["potential"]
e_dmrg = data["energy"]
x_dmrg = data["magnetization_x"]

# %%
plt.plot(z_dmrg)
plt.plot(dens_f_dmrg)
plt.plot(pot)
plt.show()
# %%

for i in trange(10):

    (
        file_name,
        e,
        h,
        z,
        f_dens,
        f,
        x,
    ) = transverse_ising_sparse_simulator_sample(
        l=16,
        h_max=2.71,
        hs=pot[i],
        j1=j1,
        j2=-1,
        pbc=True,
        file_name="nothing",
        check_2nn=False,
        eps_breaking=eps_breaking
    )

    if i == 0:
        e_quspin = e
        hs = h
        z_quspin = z
        fs_dens_quspin = f_dens
        f_quspin = f
        x_quspin = x
    else:
        e_quspin = np.append(e_quspin, e)
        hs = np.append(hs, h, axis=0)
        z_quspin = np.append(z_quspin, z, axis=0)
        fs_dens_quspin = np.append(fs_dens_quspin, f_dens, axis=0)
        f_quspin = np.append(f_quspin, f)
        x_quspin = np.append(x_quspin, x, axis=0)

# %% exact diagonalization
# hs = pot
# for r in trange(10):
#     device = "cpu"
#     qs = QuantumSpinSystem(l=l, device=device)
#     # hamiltonian
#     hamiltonian = torch.zeros((2 ** l, 2 ** l), device=device)
#     for k in range(l):
#         hamiltonian = (
#             hamiltonian
#             + j1 * torch.matmul(qs.s_x[k], qs.s_x[(k + 1) % l])
#             + hs[r, k] * qs.s_z[k] + eps_breaking*qs.s_x[k]
#         )

#     e, psi = torch.linalg.eigh(hamiltonian)

#     m_z = []
#     m_x = []
#     for i in range(l):
#         m_z.append(
#             torch.einsum(
#                 "i,ij,j->", torch.conj(psi[:, 0]
#                                        ), qs.s_z[i].to(torch.float), psi[:, 0]
#             )
#             .detach()
#             .cpu()
#             .numpy()
#         )
#         m_x.append(
#             torch.einsum(
#                 "i,ij,j->", torch.conj(psi[:, 0]
#                                        ), qs.s_x[i].to(torch.float), psi[:, 0]
#             )
#             .detach()
#             .cpu()
#             .numpy()
#         )
#     m_z = np.asarray(m_z)
#     m_x = np.asarray(m_x)

#     t_xx = []
#     for i in range(l):
#         t_xx.append(
#             -1
#             * torch.einsum(
#                 "i,ij,jk,k->",
#                 torch.conj(psi[:, 0]),
#                 qs.s_x[i].to(torch.float),
#                 qs.s_x[(i + 1) % l].to(torch.float),
#                 psi[:, 0],
#             )
#             .detach()
#             .cpu()
#             .numpy()
#             # + torch.einsum(
#             #     "i,ij,jk,k->",
#             #     torch.conj(psi[:, 0]),
#             #     qs.s_z[i].to(torch.float),
#             #     qs.s_z[(i + 2) % l].to(torch.float),
#             #     psi[:, 0],
#             # )
#             # .detach()
#             # .cpu()
#             # .numpy()
#         )
#     t_xx = np.asarray(t_xx)

#     if r == 0:
#         t_xx_ed = t_xx.reshape(1, -1)
#         z_ed = m_z.reshape(1, -1)
#         x_ed = m_x.reshape(1, -1)
#     else:
#         t_xx_ed = np.append(t_xx_ed, t_xx.reshape(1, -1), axis=0)
#         z_ed = np.append(z_ed, m_z.reshape(1, -1), axis=0)
#         x_ed = np.append(x_ed, m_x.reshape(1, -1), axis=0)
# %%
for i in range(10):

    plt.plot(z_dmrg[i], label="dmrg", linestyle="--", color="green")
    #plt.plot(z_ed[i], label="ed", linestyle=":", color="blue")
    plt.plot(z_quspin[i], label="quspin", color="red")
    plt.legend()
    plt.show()
# %%
for i in range(10):

    plt.plot(dens_f_dmrg[i], label="dmrg", linestyle="--", color="green")
    plt.plot(t_xx_ed[i], label="ed", linestyle=":", color="blue")
    plt.plot(fs_dens_quspin[i], label="quspin", color="red")
    plt.legend()
    plt.show()
# %%
for i in range(10):

    plt.plot(x_dmrg[i], label="dmrg", linestyle="--", color="green")
    #plt.plot(x_ed[i], label="ed", linestyle=":", color="blue")
    plt.plot(x_quspin[i], label="quspin", color="red")
    plt.legend()
    plt.show()
# %%
