import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

os.environ[
    "KMP_DUPLICATE_LIB_OK"
] = "True"  # uncomment this line if omp error occurs on OSX for python 3
os.environ["OMP_NUM_THREADS"] = str(
    3
)  # set number of OpenMP threads to run in parallel
os.environ["MKL_NUM_THREADS"] = str(3)  # set number of MKL threads to run in parallel

from src.utils_sparse_diag import (
    transverse_ising_sparse_DFT,
    transverse_ising_sparse_DFT_lanczos_method,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_dataset",
    type=int,
    help="# of istances in the dataset (default=150000)",
    default=15000,
)
parser.add_argument("--l", type=int, help="size of the chain ", default=16)
parser.add_argument(
    "--j1",
    type=float,
    help="the coupling costant of the spin-spin interaction (default=1)",
    default=-1.0,
)
parser.add_argument(
    "--j2",
    type=float,
    help="the coupling costant of the spin-spin interaction (default=1)",
    default=-1.0,
)

parser.add_argument(
    "--h_max",
    type=float,
    help="the maximum value of the transverse magnetic field (default=e)",
    default=2 * np.e,
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
    default="unet_periodic",
)

parser.add_argument(
    "--seed",
    type=int,
    help="seed for numpy and pytorch (default=42)",
    default=42,
)


args = parser.parse_args()
np.random.seed(args.seed)
print("n_dataset=", args.n_dataset)
hs = np.random.uniform(0, args.h_max, size=(args.n_dataset, args.l))
if not (args.lanczos):
    file_name, hs, zs, fs_dens, es = transverse_ising_sparse_DFT(
        h_max=args.h_max,
        hs=hs,
        n_dataset=args.n_dataset,
        l=args.l,
        j1=args.j1,
        j2=args.j2,
        z_2=args.z2,
        file_name=args.file_name,
        pbc=args.pbc,
        check_2nn=args.check_2nn,
        eps_breaking=args.eps_breaking,
    )
else:
    file_name, hs, zs, fs_dens, es = transverse_ising_sparse_DFT_lanczos_method(
        h_max=args.h_max,
        hs=hs,
        n_dataset=args.n_dataset,
        l=args.l,
        j1=args.j1,
        j2=args.j2,
        z_2=args.z2,
        file_name=args.file_name,
        pbc=args.pbc,
        check_2nn=args.check_2nn,
        eps_breaking=args.eps_breaking,
        dimension=args.dimension,
    )
np.savez(
    file_name,
    potential=hs,
    density=zs,
    density_F=fs_dens,
    energy=es,
)


# # %%

# data = np.load("data/dataset_dmrg/l_64_h_2.71_ndata_10.npz")
# x = data["magnetization_x"]

# for i in range(10):
#     plt.plot(x[i])
#     plt.show()
# # %%
