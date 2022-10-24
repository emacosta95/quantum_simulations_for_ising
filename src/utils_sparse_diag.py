# import the sparse eigensolver
import argparse
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import quspin
import torch
from quspin.basis import spin_basis_1d  # Hilbert space spin basis
from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.operators import quantum_LinearOperator, quantum_operator
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from tqdm.notebook import trange


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
