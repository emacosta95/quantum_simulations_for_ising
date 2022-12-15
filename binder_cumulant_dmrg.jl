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
linkdims=200
init_bonddim=384
sweep=20
n=[16,64,128,256,384]
j_coupling=-1.
omega=0.
hmaxs=[2*exp(1)-0.5,2*exp(1)-0.25,2*exp(1),2*exp(1)+0.25,2*exp(1)+0.5]
#hmaxs=LinRange(0.1,12.,nlinspace) # for studying the phase transition
ndata=100
nreplica=1
two_nn=true
pbc=true
set_noise=false



# we need to understand
# how to implement a string
# format

#fix the seed
Random.seed!(seed)
# different sizes
for k=1:length(n)
        sites=siteinds("S=1/2",n[k]) #fix the basis representation
        m_4=OpSum()
        for i=1:n[k]
            for r=1:n[k]
                for t=1:n[k]
                    for y=1:n[k]
                m_4+=1.,"Sx",i,"Sx",r,"Sx",t,"Sx",y
                    end
                end
            end
        end
        m_4=MPO(m_4,sites)
        psi0=randomMPS(sites,init_bonddim) #initialize the product state
        for j=1:length(hmaxs)
                #name file
                namefile="data/check_the_2nn_phase_transition/151222/binder_cumulant_2nn_$(n[k])_l_$(hmaxs[j])_h_$(ndata)_n.npz"
                u=zeros(Float64,(ndata))
                for i=tqdm(1:ndata)

                        # initialize the field
                        h=rand(Uniform(0.,hmaxs[j]),n[k])
                        h=hmaxs[j]*ones(n[k])
                        u[i]=dmrg_nn_ising_binder_cumulant(linkdims,sweep,n[k],j_coupling,j_coupling,omega,hmaxs[j],two_nn,pbc,h,nreplica,set_noise,psi0,sites,m_4)
                        


                        #save
                        npzwrite(namefile, Dict(
                        "u"=>u))
                end
        end
end


