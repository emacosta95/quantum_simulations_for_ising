using ITensors
using Distributions
using LinearAlgebra
using Random
using Plots
using NPZ
using ProgressBars

function dmrg_nn_ising(linkdims::Int64,sweep::Int64,n::Int64,j_1::Float64,j_2::Float64,h_max::Float64,two_nn::Bool,hs::Array{Float64},pbc::Bool,psi0::ITensors.MPS,sites::Vector{Index{Int64}})

    #fix the seed
    #Random.seed!(seed)

    #fix the representation
    #define the universal part of the hamiltonian
    ham_0=OpSum()
    for i=1:n-2
        ham_0+=4*j_1,"Sx",i,"Sx",i+1 #1 nearest neighbours
        if two_nn
            ham_0+=4*j_2,"Sx",i,"Sx",i+2 #2 nearest neighbours
        end
    #    h_i=rand(Uniform(0,h_max))
    #    push!(potential,h_i)
    #    hamiltonian+=h_i,"Sz",i # external random field
    end
    ham_0+=4*j_1,"Sx",n-1,"Sx",n #1 nearest neighbours
    if pbc #if periodic boundary condition
        if two_nn
            ham_0+=4*j_2,"Sx",n-1,"Sx",1 #2 nearest neighbours
        end
        ham_0+=4*j_1,"Sx",n,"Sx",1 #1 nearest neighbours
        if two_nn
            ham_0+=4*j_2,"Sx",n,"Sx",2 #2 nearest neighbours
        end
    end

    for j=1:n
        ham_0+=2*hs[j],"Sz",j # external random field
    end

    # for j=1:n
    #     ham_ext+=2*eps_breaking,"Sx",j # to
    # end 
    #ham_ext+=2*eps_breaking,"Sx",1
    #fix the invariance problem
    #end
    # collect the external
    # field
    #push!(v_tot,potential)

    hamiltonian=ham_0

    # initialize the hamiltonian
    # as matrix product operator
    h=MPO(hamiltonian,sites) #define the hamiltonian
    

    #fix the sweeps
    sweeps = Sweeps(sweep)
    setmaxdim!(sweeps,10,20,40,50,linkdims)
    setcutoff!(sweeps, 1E-10)
    #noise!(sweeps,1E-06,1E-07,1E-08,1E-11,0)

    #compute the initial energy value it should be the same
    print("initial energy=",inner(psi0',h,psi0),"\n")


    # energy values
    energy, psi = dmrg(h,psi0, sweeps,outputlevel=1)

    #compute the transverse magnetization and the density functional per site 
    z=2*expect(psi,"Sz")
    x=2*expect(psi,"Sx")
    
    xx=4*correlation_matrix(psi,"Sx","Sx")
    x_1nn=zeros(Float64,(n))
    x_2nn=zeros(Float64,(n))   
    for j=1:n
        if j==n 
            # push!(x_1nn,xx[j,1])
            # push!(x_,xx[j,2])
            x_1nn[j]=xx[j,1]
            x_2nn[j]=xx[j,2]
        elseif j==n-1
            # push!(x_1nn,xx[j,j+1])
            # push!(x_2nn,xx[j,1])
            x_1nn[j]=xx[j,j+1]
            x_2nn[j]=xx[j,1]
        else
            # push!(x_1nn,xx[j,j+1])
            # push!(x_2nn,xx[j,j+2])
            x_1nn[j]=xx[j,j+1]
            x_2nn[j]=xx[j,j+2]
        end
    end
    if two_nn
        dens_f=j_coupling*(x_1nn+x_2nn)
    else
        dens_f=j_coupling*x_1nn
    end
    f=energy/n-dot(hs,z)/n
    #print(length(f))

    #print("type of energy/n ",typeof(h),"\n")    
    # alternative method
    return energy/n, hs,z,x,dens_f,f,xx
end



function dmrg_nn_ising_check_h_k_map(linkdims::Int64,sweep::Int64,n::Int64,j_1::Float64,j_2::Float64,h_max::Float64,two_nn::Bool,hs::Array{Float64},psi0::ITensors.MPS,sites::Vector{Index{Int64}})

    
    ham_0=OpSum()
    
    for i=1:n-2
        ham_0+=4*j_1,"Sx",i,"Sx",i+1 #1 nearest neighbours
        if two_nn
            ham_0+=4*j_2,"Sx",i,"Sx",i+2 #2 nearest neighbours
        end
    #    h_i=rand(Uniform(0,h_max))
    #    push!(potential,h_i)
    #    hamiltonian+=h_i,"Sz",i # external random field
    end
    ham_0+=4*j_1,"Sx",n-1,"Sx",n #1 nearest neighbours
    if two_nn
        ham_0+=4*j_2,"Sx",n-1,"Sx",1 #2 nearest neighbours
    end
    ham_0+=4*j_1,"Sx",n,"Sx",1 #1 nearest neighbours
    if two_nn
        ham_0+=4*j_2,"Sx",n,"Sx",2 #2 nearest neighbours
    end

    #external potential
    ham_ext=OpSum()
    for j=1:n
        ham_ext+=2*hs[j],"Sz",j # external random field
    end

    for j=1:n
        ham_ext+=2*eps_breaking,"Sx",j # to
    end 
    #ham_ext+=2*eps_breaking,"Sx",1
    #fix the invariance problem
    #end
    # collect the external
    # field
    #push!(v_tot,potential)

    hamiltonian=ham_0+ham_ext

    # initialize the hamiltonian
    # as matrix product operator
    h=MPO(hamiltonian,sites) #define the hamiltonian
    

    #fix the sweeps
    sweeps = Sweeps(sweep)
    setmaxdim!(sweeps,5,10,20,linkdims)
    setcutoff!(sweeps, 1E-6)

    # energy values
    energy, psi = dmrg(h,psi0, sweeps,outputlevel=0)

    #compute the transverse magnetization and the density functional per site 
    z=2*expect(psi,"Sz")
    # correlation
    zz=4*correlation_matrix(psi,"Sz","Sz")
    return z,zz
end