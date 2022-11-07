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
linkdims=[10,20,30,40,50,60,70,80]
sweep=[10,20,30,40,50,60,70,80]
n=128
j_coupling=-1.
hmaxs=2*exp(1)
#hmaxs=LinRange(0.1,12.,nlinspace) # for studying the phase transition
ndata=1
two_nn=true
pbc=false
eps_breaking=0.
#namefile="data/dataset_dmrg/l_{}_h_{}_ndata_{}".format(n,h_max,ndata)
# we need to understand
# how to implement a string
# format

#fix the seed
Random.seed!(seed)

# different sizes
for r=1:8
    for j=1:8
        #name file
        namefile="data/check_dmrg/test_unet_periodic_2nn_obc_$(n)_l_$(hmaxs)_h_$(ndata)_n_$(sweep[r])_sweep_$(linkdims[j])_bonddim.npz"
        e_tot = zeros(Float64,(ndata))
        v_tot = zeros(Float64,(ndata,n))
        f_tot=zeros(Float64,(ndata))
        dens_f_tot= zeros(Float64,(ndata,n))
        z_tot= zeros(Float64,(ndata,n))
        x_tot=zeros(Float64,(ndata,n))
        xxs=zeros(Float64,(ndata,n,n))
        for i=tqdm(1:ndata)

                # initialize the field
                h=rand(Uniform(0.,hmaxs),n[j])
                energy,potential,z,x,dens_f,f,xx=dmrg_nn_ising(seed,linkdims[j],sweep[i],n,j_coupling,j_coupling,hmaxs,eps_breaking,namefile,two_nn,h,pbc)
                
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
end