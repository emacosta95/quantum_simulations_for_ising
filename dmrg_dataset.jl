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
n=[(i)*16 for i=1:10]
j_coupling=-1.
hmaxs=5.44
#hmaxs=LinRange(0.1,12.,nlinspace) # for studying the phase transition
ndata=100
two_nn=true
pbc=true
nreplica=5
#namefile="data/dataset_dmrg/l_{}_h_{}_ndata_{}".format(n,h_max,ndata)
# we need to understand
# how to implement a string
# format

#fix the seed
Random.seed!(seed)

# different sizes
for j=1:length(n)

        #name file
        namefile="data/dataset_2nn/test_dataset_191122/test_unet_periodic_2nn_$(n[j])_l_$(hmaxs)_h_$(ndata)_n.npz"
        e_tot = zeros(Float64,(ndata))
        v_tot = zeros(Float64,(ndata,n[j]))
        f_tot=zeros(Float64,(ndata))
        dens_f_tot= zeros(Float64,(ndata,n[j]))
        z_tot= zeros(Float64,(ndata,n[j]))
        x_tot=zeros(Float64,(ndata,n[j]))
        xxs=zeros(Float64,(ndata,n[j],n[j]))
        for i=tqdm(1:ndata)

                # initialize the field
                h=rand(Uniform(0.,hmaxs),n[j])
                energy,potential,z,x,dens_f,f,xx=dmrg_nn_ising_composable(linkdims,sweep,n[j],j_coupling,j_coupling,hmaxs,two_nn,h,pbc,nreplica)
                
                # cumulate
                e_tot[i]=energy
                v_tot[i,:]=potential
                z_tot[i,:]=z
                x_tot[i,:]=x
                dens_f_tot[i,:]=dens_f
                f_tot[i]=f
                xxs[i,:,:]=xx

                #save
                npzwrite(namefile, Dict("density" => z_tot, "energy" => e_tot, "F" => f_tot,"density_F"=> dens_f_tot,"potential"=>v_tot,"magnetization_x"=>x_tot,
                "correlation"=>xxs))
        end
end