# import
include("src/utils_dmrg.jl")

# parameters
seed=125
linkdims=30
n=32
j_coupling=-1.
h_max=2.71
ndata=100
two_nn=true
eps_breaking=10^(-2)
#namefile="data/dataset_dmrg/l_{}_h_{}_ndata_{}".format(n,h_max,ndata)
# we need to understand
# how to implement a string
# format
namefile=raw"data/dmrg_2nn/l_$n_h_$h_max_ndata_$ndata.npz"
print(namefile)
dmrg_nn_ising(seed,linkdims,n,j_coupling,j_coupling,h_max,ndata,eps_breaking,namefile,two_nn)