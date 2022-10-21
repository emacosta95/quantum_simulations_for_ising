# import
using LinearAlgebra
include("src/utils_dmrg.jl")
using ITensors

BLAS.set_num_threads(10)
# parameters
seed=125
linkdims=30
sweep=30
n=[8,16,32,64,128]
j_coupling=-1.
hmaxs=0
nlinspace=5
#hmaxs=LinRange(0.1,12.,nlinspace) # for studying the phase transition
ndata=10
two_nn=false
eps_breaking=10^(-2)
#namefile="data/dataset_dmrg/l_{}_h_{}_ndata_{}".format(n,h_max,ndata)
# we need to understand
# how to implement a string
# format

for i=1:nlinspace
    namefile="data/den2magn_dataset_1nn/test_unet_periodic_1nn_l_$(n[i])_h_$(hmaxs)_ndata_$(ndata).npz"
    dmrg_nn_ising(seed,linkdims,sweep,n[i],j_coupling,j_coupling,hmaxs[i],ndata,eps_breaking,namefile,two_nn)
end