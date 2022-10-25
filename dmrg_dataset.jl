# import
using LinearAlgebra
include("src/utils_dmrg.jl")
using ITensors

BLAS.set_num_threads(10)
# parameters
seed=125
linkdims=40
sweep=30
n=[16,32,64,128,256]
j_coupling=-1.
hmaxs=7.2
nlinspace=5
#hmaxs=LinRange(0.1,12.,nlinspace) # for studying the phase transition
ndata=100
two_nn=true
eps_breaking=0.
#namefile="data/dataset_dmrg/l_{}_h_{}_ndata_{}".format(n,h_max,ndata)
# we need to understand
# how to implement a string
# format

for i=1:nlinspace
    namefile="data/dataset_2nn/test_unet_periodic_2nn_$(n[i])_l_$(hmaxs)_h_$(ndata)_n.npz"
    dmrg_nn_ising(seed,linkdims,sweep,n[i],j_coupling,j_coupling,hmaxs,ndata,eps_breaking,namefile,two_nn)
end