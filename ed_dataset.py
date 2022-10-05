from src.utils_sparse_diagonalization import transverse_ising_sparse_simulator
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
    "--n_dataset", type=int, help="# of istances in the dataset (default=150000)", default=15000
)
parser.add_argument(
    "--l", type=int, help="size of the chain ", default=16
)
parser.add_argument(
    "--j1",
    type=float,
    help="the coupling costant of the spin-spin interaction (default=1)",
    default=-1.,
)
parser.add_argument(
    "--j2",
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
hs=np.random.uniform(0,args.h_max,size=(args.n_dataset,args.l))
file_name,es,hs,zs,fs_dens,fs,xs=transverse_ising_sparse_simulator(h_max=args.h_max,hs=hs,n_dataset=args.n_dataset,l=args.l,j1=args.j1,j2=args.j2,z_2=args.z2,file_name=args.file_name,pbc=args.pbc,check_2nn=args.check_2nn)
np.savez(file_name,energy=es,potential=hs,density=zs,density_F=fs_dens,F=fs,magnetization_x=xs)