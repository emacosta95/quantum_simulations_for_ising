import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.fft import fft, ifft
from torch import conj
from tqdm import trange

from src.utils import (nambu_diagonalization_ising_model,
                       parallel_nambu_correlation_ising_model)

# quantum transverse ising model 1d
torch.manual_seed(42)
torch.set_num_threads(10)


parser = argparse.ArgumentParser()

parser.add_argument(
    "--n_dataset", type=int, help="# of istances in the dataset (default=150000)", default=150000
)
parser.add_argument(
    "--nbatch", type=int, help="# batches for the parallel computation (default=150)", default=150
)
parser.add_argument(
    "--l", type=int, help="size of the chain ", default=64
)
parser.add_argument(
    "--j",
    type=float,
    help="the coupling costant of the spin-spin interaction (default=1)",
    default=-1.,
)
parser.add_argument(
    "--h_max",
    type=float,
    help="the maximum value of the transverse magnetic field (default=e)",
    default=np.e,
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


hs = np.random.uniform(0, args.h_max, size=(args.n_dataset, args.l))

ss_x, _, s_z = parallel_nambu_correlation_ising_model(
    nbatch=args.nbatch, l=args.l, j_coupling=args.j, hs=hs, device='cpu', name_file='none', pbc=args.pbc)


pbc_name = ''
if args.pbc:
    pbc_name = 'pbc_'

z2_name = ''
if args.z2:
    ss_x = np.append(ss_x, ss_x, axis=0)
    s_z = np.append(s_z, -1*s_z, axis=0)
    indices = np.arange(s_z.shape[0])
    np.random.shuffle(indices)
    ss_x = ss_x[indices]
    s_z = s_z[indices]
    z2_name = 'augmentation'

name = args.file_name+f'_h_{args.h_max}_'+f'n_{args.n_dataset}_' + \
    f'l_{args.l}_'+pbc_name+f'j_{-1*args.j}'+z2_name
np.savez('data/correlation_1nn/'+name, density=s_z,
         correlation=ss_x, potential=hs)
