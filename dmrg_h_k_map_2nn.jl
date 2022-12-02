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
seed=425
linkdims=200
init_bonddim=512
sweep=20
n=[256,512]
j_coupling=-1.
#hmaxs=[2*exp(1)-0.5,2*exp(1)-0.3,2*exp(1)-0.1,2*exp(1),2*exp(1)+0.1,2*exp(1)+0.3,2*exp(1)+0.5]
hmaxs=LinRange(0.1,12.,20) # for studying the phase transition
ndata=100
two_nn=true
pbc=true
set_noise=false
nreplica=5
omega=0.
# we need to understand
# how to implement a string
# format

#fix the seed
Random.seed!(seed)
# different sizes
for k=1:length(n)
        sites=siteinds("S=1/2",n[k]) #fix the basis representation
        psi0=randomMPS(sites,init_bonddim) #initialize the product state
        for j=1:length(hmaxs)
                #name file
                namefile="data/dmrg_h_k_map_2nn/dataset_011222/h_k_check_2nn_$(n[k])_l_$(hmaxs[j])_h_$(ndata)_n.npz"
                v_tot = zeros(Float64,(ndata,n[k]))
                z_tot= zeros(Float64,(ndata,n[k]))
                zzs=zeros(Float64,(ndata,n[k],n[k]))
                for i=tqdm(1:ndata)

                        # initialize the field
                        h=rand(Uniform(0.,hmaxs[j]),n[k])
                        z,zz=dmrg_nn_ising_check_h_k_map(linkdims,sweep,n[k],j_coupling,j_coupling,omega,hmaxs[j],two_nn,pbc,h,nreplica,set_noise,psi0,sites)
                        
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