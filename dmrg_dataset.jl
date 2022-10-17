# import
using LinearAlgebra
include("src/utils_dmrg.jl")
using ITensors

BLAS.set_num_threads(10)
# parameters
seed=125
linkdims=30
sweeps=10
n=64
j_coupling=-1.
hmax=2.71
ndata=10
two_nn=false
eps_breaking=10^(-2)
#namefile="data/dataset_dmrg/l_{}_h_{}_ndata_{}".format(n,h_max,ndata)
# we need to understand
# how to implement a string
# format
namefile=raw"data/den2magn_dataset_1nn/test_unet_periodic_1nn_l_16_h_2.71_ndata_100.npz"
dmrg_nn_ising(seed,linkdims,sweeps,n,j_coupling,j_coupling,hmax,ndata,eps_breaking,namefile,two_nn)