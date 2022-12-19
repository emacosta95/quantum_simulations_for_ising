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

sweep=20
n=[8,10,12,14,16]
j_coupling=-1.
omega=0.
hmaxs=[2*exp(1)-0.5,2*exp(1)-0.25,2*exp(1),2*exp(1)+0.25,2*exp(1)+0.5]
hmaxs=[2.0,2.25,2.5,2.6,2.71,2.75,3.0,3.25,3.5,3.75,4.0,4.25,4.5,4.75,5.0,5.25,5.44,5.5,5.75,6.0,6.25,6.5,6.75,7.0]
#hmaxs=LinRange(0.5,1.5,100)
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
        print("initialize sites.. \n")
        sites=siteinds("S=1/2",n[k]) #fix the basis representation
        print("initialize the MPO m^4... \n")
        m_4=OpSum()
        m_2=OpSum()
        for i=1:n[k]
            for r=1:n[k]
                m_2+=(4.),"Sx",i,"Sx",r
                for t=1:n[k]
                    for y=1:n[k]
                        m_4+=((2.))^4,"Sx",i,"Sx",r,"Sx",t,"Sx",y
                    end
                end
            end
        end
        m_4=MPO(m_4,sites)
        m_2=MPO(m_2,sites)
        print("initialize the random state... \n")
        init_bonddim=n[k]
        psi0=randomMPS(sites,init_bonddim) #initialize the product state
        for j=1:length(hmaxs)
                #name file
                namefile="data/check_the_2nn_phase_transition/151222/binder_cumulant_obc_2nn_$(n[k])_l_$(hmaxs[j])_h_$(ndata)_n.npz"
                u_4=zeros(Float64,(ndata))
                u_2=zeros(Float64,(ndata))
                u=zeros(Float64,(ndata))
                for i=tqdm(1:ndata)

                        # initialize the field
                        h=rand(Uniform(0.,hmaxs[j]),n[k])
                        u_4[i],u_2[i],u[i]=dmrg_nn_ising_binder_cumulant(linkdims,sweep,n[k],j_coupling,j_coupling,omega,hmaxs[j],two_nn,pbc,h,nreplica,set_noise,psi0,sites,m_4,m_2)
                        
                        #save
                        npzwrite(namefile, Dict(
                        "m4"=>u_4,"m2"=>u_2,"u"=>u))
                end
        end
end


