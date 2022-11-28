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
linkdims=100
sweep=20
n=[16,24,32,64,96,128,136,256]
j_coupling=-1.
hmaxs=[2*exp(1)-0.5,2*exp(1)-0.3,2*exp(1)-0.1,2*exp(1),2*exp(1)+0.1,2*exp(1)+0.3,2*exp(1)+0.5]
#hmaxs=LinRange(0.1,12.,nlinspace) # for studying the phase transition
ndata=100
two_nn=true
pbc=true
nreplica=5
# we need to understand
# how to implement a string
# format

#fix the seed
Random.seed!(seed)
# different sizes
for k=1:length(n)
        for j=1:length(hmaxs)
                #name file
                namefile="data/dmrg_h_k_map_2nn/h_k_check_2nn_$(n[k])_l_$(hmaxs[j])_h_$(ndata)_n.npz"
                v_tot = zeros(Float64,(ndata,n[k]))
                z_tot= zeros(Float64,(ndata,n[k]))
                zzs=zeros(Float64,(ndata,n[k],n[k]))
                for i=tqdm(1:ndata)

                        # initialize the field
                        h=rand(Uniform(0.,hmaxs[j]),n[k])
                        z,zz=dmrg_nn_ising_check_h_k_map(linkdims,sweep,n[k],j_coupling,j_coupling,omega,hmaxs[j],two_nn,pbc,h,nreplica)
                        
                        # cumulate
                        for g=1:n[k]
                                v_tot[i,g]=h[g]
                                z_tot[i,g]=z[g]
                                for r=1:n[k]
                                        zzs[i,g,r]=zz[g,r]
                                end
                        end

                        #save
                        npzwrite(namefile, Dict("potential"=>v_tot,"density"=>z_tot,
                        "correlation"=>zzs))
                end
        end
end