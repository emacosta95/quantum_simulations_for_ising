#%% Sketch for an efficient way of computing the adjency matrix
from typing import List, Callable, Tuple
import numpy as np
import quspin
from quspin.operators import hamiltonian, quantum_LinearOperator
from quspin.basis import spin_basis_1d
import os

os.environ[
    "KMP_DUPLICATE_LIB_OK"
] = "True"  # uncomment this line if omp error occurs on OSX for python 3
os.environ["OMP_NUM_THREADS"] = str(
    5
)  # set number of OpenMP threads to run in parallel
os.environ["MKL_NUM_THREADS"] = str(5)  # set number of MKL threads to run in parallel


def compute_correlation(psi: np.array, l: int, basis: quspin.basis, direction: str):
    for i in range(l):
        exp_m_i = []
        for j in range(l):
            coupling = [[1, i, j]]
            op = [direction, coupling]
            static = [op]
            dynamic = []
            m = quantum_LinearOperator(
                static,
                basis=basis,
                dtype=np.float64,
                check_symm=False,
                check_herm=False,
                check_pcon=False,
            )
            exp_m_i.append(m.expt_value(psi))
        exp_m_i = np.asarray(exp_m_i)

        if i == 0:
            exp_m = exp_m_i.reshape(1, -1)
        else:
            exp_m = np.append(exp_m, exp_m_i.reshape(1, -1), axis=0)

    return exp_m


def compute_magnetization(psi: np.array, l: int, basis: quspin.basis, direction: str):
    # define the connection
    m = {}
    exp_m = []
    for i in range(l):
        coupling = [[1, i]]
        op = [direction, coupling]
        static = [op]
        dynamic = []
        m = quantum_LinearOperator(
            static,
            basis=basis,
            dtype=np.float64,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
        )
        exp_m.append(m.expt_value(psi))
    return exp_m


def density_functional_adj_matrix(
    psi: np.array, n: int, basis: quspin.basis, adj_matrix: List[List], direction: str
):
    m = {}
    exp_m = []
    for i in range(n):
        coupling = [adj_matrix[i]]
        op = [direction, coupling]
        static = [op]
        dynamic = []
        m = quantum_LinearOperator(
            static,
            basis=basis,
            dtype=np.float64,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
        )
        exp_m.append(m.expt_value(psi))
    return exp_m


# coupling function set
def first_nn2d_pbc(i: int, n: int, j_value: float) -> List:
    """define the coupling function with J in the 1nn Ising 2d square lattice

    Args:
        i (int): index
        n (int): number of sites
        j_value (float): float value of the coupling costant

    Returns:
        List: indices coupled to i-th
    """
    jdx = []
    j_value_i = [j_value for k in range(2)]
    # nearest neighbour  pbc condition
    jdx.append(int(i / int(np.sqrt(n))) * int(np.sqrt(n)) + (i + 1) % int(np.sqrt(n)))
    jdx.append((i + int(np.sqrt(n))) % n)
    return jdx, j_value_i


def first_nn2d(i: int, n: int, j_value: float) -> List:
    """define the coupling function with J in the 1nn Ising 2d square lattice

    Args:
        i (int): index
        n (int): number of sites
        j_value (float): float value of the coupling costant

    Returns:
        List: indices coupled to i-th
    """
    jdx = []

    j_value_i = [j_value for k in range(2)]
    # nearest neighbour  pbc condition
    if (i + 1) % int(np.sqrt(n)) != 0:
        jdx.append(
            int(i / int(np.sqrt(n))) * int(np.sqrt(n)) + (i + 1) % int(np.sqrt(n))
        )
    if i + int(np.sqrt(n)) < n:
        jdx.append((i + int(np.sqrt(n))) % n)

    return jdx, j_value_i


def get_adj_matrix(n: int, sigma: Callable, j_value) -> List[List]:
    j_coupling = []
    for i in range(n):
        jdx, j_value_i = sigma(i, n, j_value)
        # print(f"index i={i} with nn={jdx}")
        for r, j in enumerate(jdx):
            # index, index, value
            j_coupling.append([j_value_i[r], i, j])

    return j_coupling


def transverse_ising_sparse_simulator_2d(
    h_max: int,
    hs: np.ndarray,
    n: int,
    j: float,
    delta: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    # file_name information
    text_z2 = ""
    text_field = f"{h_max:.2f}_h"

    hs = hs
    # the basis of the representation
    basis = spin_basis_1d(n)

    # the coupling terms

    h_z = [[hs[k], k] for k in range(n)]  # external field
    h_x = [[-1 * delta, k] for k in range(n)]
    adj_matrix = get_adj_matrix(n, first_nn2d_pbc, j_value=j)
    j_coupling = [jcoupling for jcoupling in adj_matrix]
    # print(j_coupling)
    # implement the static part of the hamiltonian
    static = [["xx", j_coupling], ["x", h_x], ["z", h_z]]
    dynamic = []
    ham = hamiltonian(
        static,
        dynamic,
        basis=basis,
        dtype=np.float64,
        check_symm=False,
        check_herm=False,
        check_pcon=False,
    )
    e, psi_0 = ham.eigsh(
        k=2,
        which="BE",
        maxiter=1e4,
    )
    e = e[0]
    psi_0 = psi_0[:, 0]
    x = compute_magnetization(psi_0, l=n, basis=basis, direction="x")
    z = compute_magnetization(psi_0, l=n, basis=basis, direction="z")
    zz = compute_correlation(psi_0, l=n, basis=basis, direction="zz")
    xx = compute_correlation(psi_0, l=n, basis=basis, direction="xx")
    # print("x=", x)
    # print("z=", z)
    z = np.asarray(z)
    x = np.asarray(x)
    f_dens = density_functional_adj_matrix(
        psi_0, n=n, basis=basis, adj_matrix=adj_matrix, direction="xx"
    )
    f_dens = np.asarray(f_dens)
    f = e / n - np.average(z * hs)
    f = f
    z = z.reshape(1, int(np.sqrt(n)), int(np.sqrt(n)))
    f_dens = f_dens.reshape(1, int(np.sqrt(n)), int(np.sqrt(n)))
    e = e / n
    x = x.reshape(1, int(np.sqrt(n)), int(np.sqrt(n)))

    return e, hs, z, f_dens, f, x, zz, xx


#%%
h_maxs = np.linspace(0, 2, 20)
delta = 1
j = 1.0
n = 16
h_max = 1

for i, h_max in enumerate(h_maxs):
    print(i)
    hs = np.random.uniform(h_max, h_max + 0.0001, (n))

    e, hs, z, f_dens, f, x, zz, xx = transverse_ising_sparse_simulator_2d(
        h_max=h_max, hs=hs, n=n, j=j, delta=delta
    )
    if i == 0:
        es = e
        pot = hs.reshape(1, int(np.sqrt(n)), int(np.sqrt(n)))
        zs = z
        fs_dens = f_dens
        xs = x
        zzs = zz
        xxs = xx
    else:
        es = np.append(es, e)
        pot = np.append(pot, hs.reshape(1, int(np.sqrt(n)), int(np.sqrt(n))), axis=0)
        zs = np.append(zs, z, axis=0)
        fs_dens = np.append(fs_dens, f_dens, axis=0)
        xs = np.append(xs, x, axis=0)
        zzs = np.append(zzs, zz, axis=0)
        xxs = np.append(xxs, xx, axis=0)

# %%
import matplotlib.pyplot as plt

for i in range(zs.shape[0]):
    # plt.imshow(zs[i],label=f'z,h={h_maxs[i]}')
    # plt.colorbar()
    # plt.legend()
    # plt.show()
    print("energy=", es[i])
    print(h_maxs[i])
    plt.imshow(xs[i], label=f"x,h={h_maxs[i]}")
    plt.colorbar()
    plt.legend()
    plt.show()

#%%
idx = np.arange(int(np.sqrt(n)))
stag = (-1) ** (idx[None, :] + idx[:, None])
stag_magn = []
for i in range(len(xs)):
    stag_magn.append(np.average(xs[i] * stag))

plt.plot(h_maxs, np.abs(stag_magn))


# %%
n = 9

adj_matrix = get_adj_matrix(n, first_nn2d, j_value=j)

print(adj_matrix)
# %%
