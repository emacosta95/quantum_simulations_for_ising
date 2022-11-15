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
hmaxs=[5.44]
#hmaxs=LinRange(0.1,12.,nlinspace) # for studying the phase transition
ndata=100
two_nn=true
pbc=true

# we need to understand
# how to implement a string
# format

#fix the seed
Random.seed!(seed)
# different sizes
for k=1:length(n)
        sites=siteinds("S=1/2",n[k]) #fix the basis representation
        psi0=randomMPS(sites,10) #initialize the product state
        for j=1:length(hmaxs)
                #name file
                namefile="data/input_output_map/input_output_2nn_$(n[k])_l_$(hmaxs[j])_h_$(ndata)_n.npz"
                v_tot = zeros(Float64,(ndata,n[k]))
                z_tot= zeros(Float64,(ndata,n[k]))
                f_tot=zeros(Float64,(ndata,n[k]))
                zxxs=zeros(Float64,(ndata,n[k],n[k]))
                for i=tqdm(1:ndata)

                        # initialize the field
                        h=rand(Uniform(0.,hmaxs[j]),n[k])
                        zxx,z,f=dmrg_nn_ising_input_output_map(linkdims,sweep,n[k],j_coupling,j_coupling,hmaxs[j],two_nn,pbc,h,psi0,sites)
                        
                        # cumulate
                        for g=1:n[k]
                                v_tot[i,g]=h[g]
                                z_tot[i,g]=z[g]
                                f_tot[i,g]=f[g]
                                for r=1:n[k]
                                        zxxs[i,g,r]=zxx[g,r]
                                end
                        end

                        #save
                        npzwrite(namefile, Dict("potential"=>v_tot,
                        "correlation"=>zxxs, "density"=>z_tot,"density_F"=>f_tot))
                end
        end
end