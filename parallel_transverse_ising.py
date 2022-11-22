# %% using a standard cycle
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.fft import fft, ifft
from torch import conj
from tqdm import trange

from src.utils import (
    nambu_diagonalization_ising_model,
    parallel_nambu_diagonalization_ising_model,
)

# quantum transverse ising model 1d
torch.manual_seed(42)
torch.set_num_threads(10)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--n_dataset",
    type=int,
    help="# of istances in the dataset (default=150000)",
    default=150000,
)
parser.add_argument(
    "--nbatch",
    type=int,
    help="# batches for the parallel computation (default=150)",
    default=150,
)
parser.add_argument(
    "--train",
    type=bool,
    help="if True, prepare the train dataset (default=True)",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument("--l", type=int, help="size of the chain ", default=16)
parser.add_argument(
    "--j",
    type=float,
    help="the coupling costant of the spin-spin interaction (default=1)",
    default=1.0,
)
parser.add_argument(
    "--h_max",
    type=float,
    nargs="+",
    help="the maximum value of the transverse magnetic field (default=e)",
    default=[np.e],
)

parser.add_argument(
    "--z2",
    type=bool,
    help="if True, augmentation using Z2 is implemented",
    action=argparse.BooleanOptionalAction,
)

parser.add_argument(
    "--pbc",
    type=bool,
    help="if True, consider the periodic boundary condition (default=True)",
    action=argparse.BooleanOptionalAction,
)
parser.add_argument(
    "--file_name",
    type=str,
    help="comments on the file name (default='periodic')",
    default="unet_periodic",
)
parser.add_argument(
    "--device",
    type=str,
    help="the threshold difference for the early stopping (default=device available)",
    default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)
parser.add_argument(
    "--seed",
    type=int,
    help="seed for numpy and pytorch (default=42)",
    default=42,
)


args = parser.parse_args()


# define both the field and the coupling vector


if args.train:
    print("train")
    n_dataset = args.n_dataset
else:
    print("valid")
    n_dataset = int(0.1 * args.n_dataset)


l = args.l
pbc = args.pbc
np.random.seed(args.seed)
j_coupling = args.j


name_file = args.file_name
h_max = args.h_max

# only now
# h_max=np.random.uniform(1,8,size=50)
for i, h in enumerate(h_max):
    print(f"h_max={h:1f}")
    hs = np.random.uniform(0, h, (n_dataset, l))
    _, ms, _, f, fm, e = parallel_nambu_diagonalization_ising_model(
        nbatch=args.nbatch,
        l=l,
        j_coupling=j_coupling,
        hs=hs,
        device=args.device,
        pbc=pbc,
    )
    if i == 0:
        hs_tot = hs
        ms_tot = ms
        fm_tot = fm
        f_tot = f
        e_tot = e

        text_field = f"{h:.1f}"
    else:
        hs_tot = np.append(hs_tot, hs, axis=0)
        ms_tot = np.append(ms_tot, ms, axis=0)
        fm_tot = np.append(fm_tot, fm, axis=0)
        f_tot = np.append(f_tot, f, axis=0)
        e_tot = np.append(e_tot, e, axis=0)
        text_field = text_field + f"_{h:.1f}"
text_field = text_field + "_h"
# only in the case of mixed h
# text_field='h_range_1-8'
text_z2 = ""
if args.z2:
    text_z2 = "_augmentation"

print(text_field)

if args.train:
    if args.z2:
        ms_tot = np.append(ms_tot, -1 * ms_tot, axis=0)
        fm_tot = np.append(fm_tot, fm_tot, axis=0)
        f_tot = np.append(f_tot, f_tot, axis=0)

    p = np.random.permutation(ms_tot.shape[0])
    ms_tot = ms_tot[p]
    fm_tot = fm_tot[p]
    f_tot = f_tot[p]
    hs_tot = hs_tot[p]
    print(f_tot.shape, ms_tot.shape)
    np.savez(
        f"data/dataset_1nn/"
        + args.file_name
        + text_z2
        + f"_{l}_l_"
        + text_field
        + f"_{ms_tot.shape[0]}_n",
        density=ms_tot,
        density_F=fm_tot,
        F=f_tot,
        potential=hs_tot,
    )
else:
    np.savez(
        f"data/dataset_1nn/"
        + args.file_name
        + f"_{l}_l_"
        + text_field
        + f"_{ms_tot.shape[0]}_n",
        density=ms_tot,
        density_F=fm_tot,
        F=f_tot,
        potential=hs_tot,
        energy=e_tot,
    )

# %%
