# import
include("src/utils_dmrg.jl")

# parameters
seed=125
linkdims=30
n=512
j_coupling=-1.
h_max=2.71
ndata=10
#namefile="data/dataset_dmrg/l_{}_h_{}_ndata_{}".format(n,h_max,ndata)
# we need to understand
# how to implement a string
# format
namefile="data/dataset_dmrg/l_512_h_2.71_ndata_10.npz"
dmrg_2nn_ising(seed,linkdims,n,j_coupling,h_max,ndata,namefile)