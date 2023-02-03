import argparse
import os
import sys

import numpy as np
from src.utils_sparse_diag import (
    ising_external_field,
    compute_magnetization,
    get_gs,
    adj_generator,
    ising_coupling,
    functional_f,
)
from quspin.basis import spin_basis_1d
from tqdm import trange
from typing import Tuple, List
import numpy as np


os.environ[
    "KMP_DUPLICATE_LIB_OK"
] = "True"  # uncomment this line if omp error occurs on OSX for python 3
os.environ["OMP_NUM_THREADS"] = str(
    3
)  # set number of OpenMP threads to run in parallel
os.environ["MKL_NUM_THREADS"] = str(3)  # set number of MKL threads to run in parallel


parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_dataset",
    type=int,
    help="# of istances in the dataset (default=15000)",
    default=15000,
)
parser.add_argument("--l", type=int, help="size of the chain ", default=16)
parser.add_argument(
    "--j",
    type=float,
    help="the coupling costant of the spin-spin interaction (default=1)",
    default=1.0,
)

parser.add_argument(
    "--alpha",
    type=float,
    help="degree of the power law decay (default=4)",
    default=4.0,
)


parser.add_argument(
    "--h_max",
    type=float,
    help="the maximum value of the transverse magnetic field (default=e)",
    default=3,
)

parser.add_argument(
    "--eps_breaking",
    type=float,
    help="the maximum value of the transverse magnetic field (default=10**-2)",
    default=0,
)

parser.add_argument(
    "--pbc",
    type=bool,
    help="if True, consider the periodic boundary condition (default=True)",
    action=argparse.BooleanOptionalAction,
)

parser.add_argument(
    "--z2",
    type=bool,
    help="if True, consider the augmentation with the z2 symmetry (default=True)",
    action=argparse.BooleanOptionalAction,
)

parser.add_argument(
    "--lanczos",
    type=bool,
    help="if True, use the lanczos algorithm (default=True)",
    action=argparse.BooleanOptionalAction,
)


parser.add_argument(
    "--dimension",
    type=int,
    help="dimension of the lanczos subspace (default=True)",
    default=50,
)

parser.add_argument(
    "--check_2nn",
    type=bool,
    help="if True, consider the 2nn Ising Model (default=True)",
    action=argparse.BooleanOptionalAction,
)

parser.add_argument(
    "--file_name",
    type=str,
    help="comments on the file name (default='periodic')",
    default="unet_long_range_pbc",
)

parser.add_argument(
    "--seed",
    type=int,
    help="seed for numpy and pytorch (default=42)",
    default=42,
)


class LongRange:
    def __init__(self, alpha: int, j0: float) -> None:
        self.alpha = alpha
        self.j0 = j0

    def long_range(self, i: int, l: int) -> Tuple[List[int], List[float]]:
        jdx = []
        values = []
        for j in range(-int(l / 2), int(l / 2)):
            jdx.append((j + i) % l)  # append the pbc index
            # the traslational invariance leads to a i-independent relation
            if j == 0:
                values.append(
                    0.5 * self.j0 / ((np.abs(j - l)) ** self.alpha)
                    + 0.5 * self.j0 / ((np.abs(j + l)) ** self.alpha)
                    + 0.5 * self.j0 / ((np.abs(j + 2 * l)) ** self.alpha)
                    + 0.5 * self.j0 / ((np.abs(j - 2 * l)) ** self.alpha)
                )
            else:
                values.append(
                    0.5 * self.j0 / ((np.abs(j)) ** self.alpha)
                    + 0.5 * self.j0 / ((np.abs(j - l)) ** self.alpha)
                    + 0.5 * self.j0 / ((np.abs(j + l)) ** self.alpha)
                    + 0.5 * self.j0 / ((np.abs(j + 2 * l)) ** self.alpha)
                    + 0.5 * self.j0 / ((np.abs(j - 2 * l)) ** self.alpha)
                )
        return jdx, values


args = parser.parse_args()
np.random.seed(args.seed)
print("n_dataset=", args.n_dataset)
hs = np.random.uniform(0, args.h_max, size=(args.n_dataset, args.l))

zs = {}
fs = {}


longrange = LongRange(alpha=args.alpha, j0=args.j)
adj = adj_generator(l=args.l, f=longrange.long_range)
ham0, f_op = ising_coupling(
    adj=adj, l=args.l, basis=spin_basis_1d(args.l), direction="zz"
)


zs = np.zeros((args.n_dataset, args.l))
funcs = np.zeros((args.n_dataset, args.l, args.l))
es = np.zeros((args.n_dataset))
for i in trange(args.n_dataset + 1):
    vx = ising_external_field(
        h=hs[i], l=args.l, basis=spin_basis_1d(args.l), direction="x"
    )
    ham = ham0 + vx
    if args.lanczos:
        e, psi = get_gs(
            ham=ham,
            eightype="Lanczos",
            lanczos_dim=args.dimension,
            basis=spin_basis_1d(args.l),
        )
    else:
        e, psi = get_gs(ham=ham, eightype="Std", basis=spin_basis_1d(args.l))
    e = e / args.l
    x = compute_magnetization(
        psi=psi, l=args.l, basis=spin_basis_1d(args.l), direction="x"
    )
    zs[i] = np.asarray(x)
    funcs[i] = functional_f(psi=psi, l=args.l, f_density_op=f_op)
    es[i] = e

    if i % 100 == 0:

        file_name = (
            "data/dataset_long_range/"
            + args.file_name
            + f"_{np.abs(args.alpha)}_alpha"
            + f"_{args.h_max}_h"
            + f"_{args.j}_j"
            + f"_{args.l}_l"
            + f"_{args.n_dataset}_n"
        )

        np.savez(
            file_name,
            potential=hs[:i],
            density=zs[:i],
            density_F=np.sum(funcs, axis=-1)[:i],
            energy=es[:i],
        )
