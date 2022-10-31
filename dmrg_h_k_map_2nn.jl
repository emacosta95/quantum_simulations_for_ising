# import
using LinearAlgebra
include("src/utils_dmrg.jl")
using ITensors
using Distributions
using Random
using ProgressBars
# fix the number of threads
BLAS.set_num_threads(10)
# parameters
seed=125
linkdims=70
sweep=30
n=16
j_coupling=-1.
hmaxs=[2*exp(1)-0.5,2*exp(1)-0.3,2*exp(1)-0.1,2*exp(1),2*exp(1)+0.1,2*exp(1)+0.3,2*exp(1)+0.5]
#hmaxs=LinRange(0.1,12.,nlinspace) # for studying the phase transition
ndata=100
two_nn=true
eps_breaking=0.
#namefile="data/dataset_dmrg/l_{}_h_{}_ndata_{}".format(n,h_max,ndata)
# we need to understand
# how to implement a string
# format

#fix the seed
Random.seed!(seed)

# different sizes
for j=1:7
        #name file
        namefile="data/dmrg_h_k_map_2nn/h_k_check_2nn_$(n)_l_$(hmaxs[j])_h_$(ndata)_n.npz"
        v_tot = zeros(Float64,(ndata,n))
        z_tot= zeros(Float64,(ndata,n))
        zzs=zeros(Float64,(ndata,n,n))
        for i=tqdm(1:ndata)

                # initialize the field
                h=rand(Uniform(0.,hmaxs[j]),n)
                z,zz=dmrg_nn_ising(seed,linkdims,sweep,n,j_coupling,j_coupling,hmaxs[j],eps_breaking,namefile,two_nn,h)
                
                # cumulate
                v_tot[i,:]=h
                z_tot[i,:].=z
                zzs[i,:,:].=zz

                #save
                npzwrite(namefile, Dict("potential"=>v_tot,"density"=>z_tot,
                "correlation"=>zzs))
        end
end