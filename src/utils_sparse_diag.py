# import the sparse eigensolver
import argparse
from typing import List, Optional, Tuple, Dict, Callable

import matplotlib.pyplot as plt
import numpy as np
import quspin
import torch
from quspin.basis import spin_basis_1d  # Hilbert space spin basis
from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.operators import quantum_LinearOperator, quantum_operator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from tqdm import trange
from quspin.tools.lanczos import lanczos_full, lanczos_iter, lin_comb_Q_T, expm_lanczos


def ising_coupling(
    adj: Dict, l: int, basis: quspin.basis, direction: str
) -> Tuple[quspin.operators.hamiltonian, Dict]:

    coupling = []
    f_density_op = {}
    for i, j in adj.keys():
        coupling.append([adj[(i, j)], i, j])
        c_density = [[adj[(i, j)], i, j]]
        c_static = [[direction, c_density]]
        op_density = quantum_LinearOperator(
            c_static,
            basis=basis,
            dtype=np.float64,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
        )
        f_density_op[(i, j)] = op_density
    static = [[direction, coupling]]
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
    return ham, f_density_op


def ising_external_field(
    h: np.ndarray, l: int, basis: quspin.basis, direction: str
) -> quspin.operators.hamiltonian:

    coupling = []
    for i in range(l):
        coupling.append([h[i], i])
    static = [[direction, coupling]]
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
    return ham


def adj_generator(l: int, f: Callable) -> Dict:

    adj = {}
    for i in range(l):
        jdx, values = f(i, l)
        for k, j in enumerate(jdx):
            adj[(i, j)] = values[k]
    return adj


def get_gs(
    ham: quspin.operators.hamiltonian,
    eightype: str,
    lanczos_dim: Optional[int] = None,
    basis: quspin.basis = None,
) -> Tuple[float, np.ndarray]:

    if eightype == "Lanczos":
        e, psi = lanczos_method(hamiltonian=ham, basis=basis, dimension=lanczos_dim)
    else:
        e, psi = ham.eigsh(k=1, which="SA")
    return e, psi


def lanczos_method(
    hamiltonian: quspin.operators.hamiltonian, basis: quspin.basis, dimension: int
):
    """Quspin Code for the Lanczos method --> http://weinbe58.github.io/QuSpin/examples/example20.html#example20-label"""
    ###### apply Lanczos
    # initial state for Lanczos algorithm
    v0 = np.random.normal(0, 1, size=basis.Ns)
    v0 = v0 / np.linalg.norm(v0)
    #
    m_GS = dimension  # Krylov subspace dimension
    #
    # Lanczos finds the largest-magnitude eigenvalues:
    e, v, q_t = lanczos_full(hamiltonian, v0, m_GS)
    #
    #
    # compute ground state vector
    psi_GS_lanczos = lin_comb_Q_T(v[:, 0], q_t)

    return e[0], psi_GS_lanczos


def functional_f(psi: np.array, l: int, f_density_op: Dict):

    f_density = np.zeros((l, l))
    for i, j in f_density_op.keys():
        f_density[i, j] = f_density_op[(i, j)].expt_value(psi)
    return f_density


def density_of_functional_pbc(
    psi: np.array, l: int, basis: quspin.basis, j_1: float, j_2: float, check_2nn: bool
):
    m = {}
    exp_m = []
    for i in range(l):

        if check_2nn:
            coupling = [[j_1, i, (i + 1) % l], [j_2, i, (i + 2) % l]]
        else:
            coupling = [[j_1, i, (i + 1) % l]]
        op = ["xx", coupling]
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


def compute_binder_cumulant(l: int, basis: quspin.basis):

    coupling_x2 = []
    coupling_x4 = []
    for i in range(l):
        for j in range(l):
            coupling_x2.append([1, i, j])
            for k in range(l):
                for q in range(l):
                    coupling_x4.append([1, i, j, k, q])
    op_x2 = ["zz", coupling_x2]
    op_x4 = ["zzzz", coupling_x4]
    static_x2 = [op_x2]
    static_x4 = [op_x4]

    mx2 = quantum_LinearOperator(
        static_x2,
        basis=basis,
        dtype=np.float64,
        check_symm=False,
        check_herm=False,
        check_pcon=False,
    )
    mx4 = quantum_LinearOperator(
        static_x4,
        basis=basis,
        dtype=np.float64,
        check_symm=False,
        check_herm=False,
        check_pcon=False,
    )

    return mx2, mx4


def transverse_ising_sparse_DFT(
    h_max: int,
    hs: np.ndarray,
    n_dataset: int,
    l: int,
    j1: float,
    j2: float,
    pbc: bool,
    z_2: bool,
    file_name: str,
    check_2nn: bool,
    eps_breaking: float,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # file_name information
    text_z2 = ""
    text_field = f"{h_max:.2f}_h"

    dir = "data/dataset_1nn/"
    if check_2nn:
        dir = "data/dataset_2nn/"

    file_name = dir + file_name + text_z2 + f"_{l}_l_" + text_field + f"_{n_dataset}_n"

    hs = hs
    # the basis of the representation
    basis = spin_basis_1d(l)

    # the coupling terms
    if pbc:
        j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
        if check_2nn:
            j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc
    else:
        j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
        if check_2nn:
            j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc

    for r in trange(n_dataset + 1):

        h = [[hs[r, k], k] for k in range(l)]  # external field
        eps_h = [[eps_breaking, k] for k in range(l)]
        if check_2nn:
            static = [["xx", j_1nn], ["xx", j_2nn], ["z", h]]  # , ["x", eps_h]]
        else:
            static = [["xx", j_1nn], ["z", h], ["x", eps_h]]
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
        e, psi_0 = ham.eigsh(k=1)  # , sigma=-1000)

        z = compute_magnetization(psi_0, l=l, basis=basis, direction="z")

        z = np.asarray(z)
        f_dens = density_of_functional_pbc(
            psi_0, l=l, basis=basis, j_1=j1, j_2=j2, check_2nn=check_2nn
        )
        f_dens = np.asarray(f_dens)
        if r == 0:
            zs = z.reshape(1, -1)
            fs_dens = f_dens.reshape(1, -1)
            es = e
        else:
            zs = np.append(zs, z.reshape(1, -1), axis=0)
            fs_dens = np.append(fs_dens, f_dens.reshape(1, -1), axis=0)
            es = np.append(es, e)

        if z_2:
            text_z2 = "_augmentation"
            fs_dens = np.append(fs_dens, fs_dens, axis=0)
            zs = np.append(zs, -1 * zs, axis=0)

        if r % 100 == 0:
            np.savez(
                file_name,
                potential=hs,
                density=zs,
                density_F=fs_dens,
                energy=es,
            )

    return file_name, hs, zs, fs_dens, es


def binder_cumulant_computation(
    h_max: int,
    hs: np.ndarray,
    n_dataset: int,
    l: int,
    j1: float,
    j2: float,
    pbc: bool,
    z_2: bool,
    file_name: str,
    check_2nn: bool,
    eps_breaking: float,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # file_name information
    text_z2 = ""
    text_field = f"{h_max:.2f}_h"

    hs = hs
    # the basis of the representation
    basis = spin_basis_1d(l)

    # the coupling terms
    if pbc:
        j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
        if check_2nn:
            j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc
    else:
        j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
        if check_2nn:
            j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc

    mx2, mx4 = compute_binder_cumulant(l=l, basis=basis)
    for r in trange(n_dataset):

        h = [[hs[r, k], k] for k in range(l)]  # external field
        eps_h = [[eps_breaking, k] for k in range(l)]
        if check_2nn:
            static = [["zz", j_1nn], ["zz", j_2nn], ["x", h]]  # , ["x", eps_h]]
        else:
            static = [["zz", j_1nn], ["x", h], ["z", eps_h]]
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
        e, psi_0 = ham.eigsh(k=1)  # , sigma=-1000)

        x2 = mx2.expt_value(psi_0)
        x4 = mx4.expt_value(psi_0)
        u = 1 - (x4 / (3 * x2 ** 2))
        if r == 0:
            us = u
            es = e
        else:
            us = np.append(us, u)
            es = np.append(es, e)

    dir = "data/dataset_1nn/"
    if check_2nn:
        dir = "data/dataset_2nn/"

    return file_name, hs, us, x2, x4


def transverse_ising_sparse_DFT_lanczos_method(
    h_max: int,
    hs: np.ndarray,
    n_dataset: int,
    l: int,
    j1: float,
    j2: float,
    pbc: bool,
    z_2: bool,
    file_name: str,
    check_2nn: bool,
    eps_breaking: float,
    dimension: int,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # file_name information
    text_z2 = ""
    text_field = f"{h_max:.2f}_h"

    hs = hs
    # the basis of the representation
    basis = spin_basis_1d(l)

    # the coupling terms
    if pbc:
        j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
        if check_2nn:
            j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc
    else:
        j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
        if check_2nn:
            j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc

    for r in trange(n_dataset):

        h = [[hs[r, k], k] for k in range(l)]  # external field
        eps_h = [[eps_breaking, k] for k in range(l)]
        if check_2nn:
            static = [["xx", j_1nn], ["xx", j_2nn], ["z", h]]  # , ["x", eps_h]]
        else:
            static = [["xx", j_1nn], ["z", h], ["x", eps_h]]
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
        # e, psi_0 = ham.eigsh(k=1)  # , sigma=-1000)
        e, psi_0 = lanczos_method(hamiltonian=ham, basis=basis, dimension=dimension)
        z = compute_magnetization(psi_0, l=l, basis=basis, direction="z")

        z = np.asarray(z)
        f_dens = density_of_functional_pbc(
            psi_0, l=l, basis=basis, j_1=j1, j_2=j2, check_2nn=check_2nn
        )
        f_dens = np.asarray(f_dens)
        if r == 0:
            zs = z.reshape(1, -1)
            fs_dens = f_dens.reshape(1, -1)
            es = e
        else:
            zs = np.append(zs, z.reshape(1, -1), axis=0)
            fs_dens = np.append(fs_dens, f_dens.reshape(1, -1), axis=0)
            es = np.append(es, e)
    if z_2:
        text_z2 = "_augmentation"
        fs_dens = np.append(fs_dens, fs_dens, axis=0)
        zs = np.append(zs, -1 * zs, axis=0)

    dir = "data/dataset_1nn/"
    if check_2nn:
        dir = "data/dataset_2nn/"

    file_name = (
        dir + file_name + text_z2 + f"_{l}_l_" + text_field + f"_{fs_dens.shape[0]}_n"
    )

    return file_name, hs, zs, fs_dens, es


def transverse_ising_sparse_Den2Magn_dataset(
    h_max: int,
    hs: np.ndarray,
    n_dataset: int,
    l: int,
    j1: float,
    j2: float,
    pbc: bool,
    z_2: bool,
    file_name: str,
    check_2nn: bool,
    eps_breaking: float,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # file_name information
    text_z2 = ""
    text_field = f"{h_max:.2f}_h"

    hs = hs
    # the basis of the representation
    basis = spin_basis_1d(l)

    # the coupling terms
    if pbc:
        j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
        if check_2nn:
            j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc
    else:
        j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
        if check_2nn:
            j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc

    for r in trange(n_dataset):

        h = [[hs[r, k], k] for k in range(l)]  # external field
        eps_h = [[eps_breaking, 0]]
        if check_2nn:
            static = [["xx", j_1nn], ["xx", j_2nn], ["z", h]]  # ["x", eps_h]]
        else:
            static = [["xx", j_1nn], ["z", h], ["x", eps_h]]
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
        e, psi_0 = ham.eigsh(k=1)  # , sigma=-1000)

        x = compute_magnetization(psi_0, l=l, basis=basis, direction="x")
        z = compute_magnetization(psi_0, l=l, basis=basis, direction="z")

        z = np.asarray(z)
        x = np.asarray(x)
        if r == 0:
            zs = z.reshape(1, -1)
            xs = x.reshape(1, -1)
        else:
            zs = np.append(zs, z.reshape(1, -1), axis=0)
            xs = np.append(xs, x.reshape(1, -1), axis=0)

    dir = "data/den2magn_dataset_1nn/"
    if check_2nn:
        dir = "data/den2magn_dataset_2nn/"

    file_name = (
        dir + file_name + text_z2 + f"_{l}_l_" + text_field + f"_{xs.shape[0]}_n"
    )

    return file_name, zs, xs


def transverse_ising_sparse_h_k_mapping_check(
    h_max: int,
    hs: np.ndarray,
    n_dataset: int,
    l: int,
    j1: float,
    j2: float,
    pbc: bool,
    z_2: bool,
    file_name: str,
    check_2nn: bool,
    eps_breaking: float,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # file_name information
    text_z2 = ""
    text_field = f"{h_max:.2f}_h"

    hs = hs
    # the basis of the representation
    basis = spin_basis_1d(l)

    # the coupling terms
    if pbc:
        j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
        if check_2nn:
            j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc
    else:
        j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
        if check_2nn:
            j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc

    for r in trange(n_dataset):

        h = [[hs[r, k], k] for k in range(l)]  # external field
        eps_h = [[eps_breaking, 0]]
        if check_2nn:
            static = [["xx", j_1nn], ["xx", j_2nn], ["z", h]]  # ["x", eps_h]]
        else:
            static = [["xx", j_1nn], ["z", h], ["x", eps_h]]
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
        e, psi_0 = ham.eigsh(k=1)  # , sigma=-1000)

        zz = compute_correlation(psi_0, l=l, basis=basis, direction="zz")
        z = compute_magnetization(psi_0, l=l, basis=basis, direction="z")
        z = np.asarray(z)
        if r == 0:
            zs = z.reshape(1, -1)
            zzs = zz.reshape(1, zz.shape[0], zz.shape[1])
        else:
            zs = np.append(zs, z.reshape(1, -1), axis=0)
            zzs = np.append(zzs, zz.reshape(1, zz.shape[0], zz.shape[1]), axis=0)

    dir = "data/den2magn_dataset_1nn/"
    if check_2nn:
        dir = "data/den2magn_dataset_2nn/"

    file_name = (
        dir + file_name + text_z2 + f"_{l}_l_" + text_field + f"_{zzs.shape[0]}_n"
    )

    return zs, zzs


def transverse_ising_sparse_h_k_mapping_check_lanczos_method(
    h_max: int,
    hs: np.ndarray,
    n_dataset: int,
    l: int,
    j1: float,
    j2: float,
    pbc: bool,
    z_2: bool,
    file_name: str,
    check_2nn: bool,
    eps_breaking: float,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # file_name information
    text_z2 = ""
    text_field = f"{h_max:.2f}_h"

    hs = hs
    # the basis of the representation
    basis = spin_basis_1d(l)

    # the coupling terms
    if pbc:
        j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
        if check_2nn:
            j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc
    else:
        j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
        if check_2nn:
            j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc

    for r in trange(n_dataset):

        h = [[hs[r, k], k] for k in range(l)]  # external field
        eps_h = [[eps_breaking, 0]]
        if check_2nn:
            static = [["xx", j_1nn], ["xx", j_2nn], ["z", h]]  # ["x", eps_h]]
        else:
            static = [["xx", j_1nn], ["z", h], ["x", eps_h]]
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
        # e, psi_0 = ham.eigsh(k=1)  # , sigma=-1000)
        e, psi_0 = lanczos_method(hamiltonian=ham, basis=basis)

        zz = compute_correlation(psi_0, l=l, basis=basis, direction="zz")
        z = compute_magnetization(psi_0, l=l, basis=basis, direction="z")
        z = np.asarray(z)
        if r == 0:
            zs = z.reshape(1, -1)
            zzs = zz.reshape(1, zz.shape[0], zz.shape[1])
        else:
            zs = np.append(zs, z.reshape(1, -1), axis=0)
            zzs = np.append(zzs, zz.reshape(1, zz.shape[0], zz.shape[1]), axis=0)

    dir = "data/den2magn_dataset_1nn/"
    if check_2nn:
        dir = "data/den2magn_dataset_2nn/"

    file_name = (
        dir + file_name + text_z2 + f"_{l}_l_" + text_field + f"_{zzs.shape[0]}_n"
    )

    return zs, zzs


def transverse_ising_sparse_simulator_sample(
    h_max: int,
    hs: np.ndarray,
    l: int,
    j1: float,
    j2: float,
    pbc: bool,
    file_name: str,
    check_2nn: bool,
    eps_breaking: float,
) -> Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # file_name information
    text_z2 = ""
    text_field = f"{h_max:.2f}_h"

    hs = hs
    # the basis of the representation
    basis = spin_basis_1d(l)

    # the coupling terms
    if pbc:
        j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
        if check_2nn:
            j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc
    else:
        j_1nn = [[j1, i, (i + 1) % l] for i in range(l)]  # pbc
        if check_2nn:
            j_2nn = [[j2, i, (i + 2) % l] for i in range(l)]  # pbc

    h = [[hs[k], k] for k in range(l)]  # external field
    eps_h = [[eps_breaking, k] for k in range(l)]
    if check_2nn:
        static = [["xx", j_1nn], ["xx", j_2nn], ["z", h]]
    else:
        static = [["xx", j_1nn], ["z", h], ["x", eps_h]]
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
    e, psi_0 = ham.eigsh(k=1)
    x = compute_magnetization(psi_0, l=l, basis=basis, direction="x")
    z = compute_magnetization(psi_0, l=l, basis=basis, direction="z")
    print("x=", x)
    print("z=", z)
    z = np.asarray(z)
    x = np.asarray(x)
    plt.plot(psi_0)
    plt.show()
    f_dens = density_of_functional_pbc(
        psi_0, l=l, basis=basis, j_1=j1, j_2=j2, check_2nn=check_2nn
    )
    f_dens = np.asarray(f_dens)
    f = e[0] / l - np.average(z * hs)
    fs = f
    zs = z.reshape(1, -1)
    fs_dens = f_dens.reshape(1, -1)
    es = e[0] / l
    xs = x.reshape(1, -1)

    return file_name, es, hs, zs, fs_dens, fs, xs
