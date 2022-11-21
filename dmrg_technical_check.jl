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
seeds=[12,35,356,145,98,236,659]
linkdims=[100,200,300,400]
sweep=range(1,20)
n=64
j_coupling=-1.
hmaxs=2*exp(1)
#hmaxs=LinRange(0.1,12.,nlinspace) # for studying the phase transition
ndata=1
two_nn=true
pbc=true
technical_check=false
eps_breaking=0.
replica=50
#namefile="data/dataset_dmrg/l_{}_h_{}_ndata_{}".format(n,h_max,ndata)
# we need to understand
# how to implement a string
# format

#fix the seed
Random.seed!(seeds[1])
#seedRNG(seed)


# initialize the field
h=rand(Uniform(0.,hmaxs),n)
#h=hmaxs*ones(n)

# different sizes
for m=1:length(seeds)
    Random.seed!(seeds[m])
    sites=siteinds("S=1/2",n) #fix the basis representation
    if technical_check
        states = [ "Dn" for k=1:n]
        psi0 = MPS(sites,states)
    else
        psi0=randomMPS(sites,10) #initialize the product state
    end

    for j=1:length(linkdims)
        for r=1:length(sweep)
        #name file
        namefile="data/check_dmrg/test_unet_periodic_2nn_$(replica)_replica_$(n)_l_$(hmaxs)_h_$(ndata)_n_$(sweep[r])_sweep_$(linkdims[j])_bonddim_$(seeds[m])_seed.npz"
        e_tot = zeros(Float64,(ndata))
        v_tot = zeros(Float64,(ndata,n))
        f_tot=zeros(Float64,(ndata))
        dens_f_tot= zeros(Float64,(ndata,n))
        z_tot= zeros(Float64,(ndata,n))
        x_tot=zeros(Float64,(ndata,n))
        xxs=zeros(Float64,(ndata,n,n))
        for i=tqdm(1:ndata)

                # energy,potential,z,x,dens_f,f,xx=dmrg_nn_ising(linkdims[j],sweep[r],n,j_coupling,j_coupling,hmaxs,two_nn,h,pbc,psi0,sites)
                #dmrg_replica
                energy,potential,z,x,dens_f,f,xx= dmrg_nn_ising_composable(linkdims[j],sweep[r],n,j_coupling,j_coupling,hmaxs,two_nn,h,pbc,5)
                


                # cumulate
                e_tot[i]=energy
                v_tot[i,:]=potential
                z_tot[i,:]=z
                x_tot[i,:]=x
                dens_f_tot[i,:]=dens_f
                f_tot[i]=f
                xxs[i,:,:]=xx
                print("energy=$(energy) \n")

                #save
                npzwrite(namefile, Dict("density" => z_tot, "energy" => e_tot, "F" => f_tot,"density_F"=> dens_f_tot,"potential"=>v_tot,"magnetization_x"=>x_tot,
                "correlation"=>xxs))
            end
        end
    end
end