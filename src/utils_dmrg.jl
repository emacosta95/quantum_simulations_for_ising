using ITensors
using Distributions
using LinearAlgebra
using Random
using Plots
using NPZ
using ProgressBars

function dmrg_nn_ising(linkdims::Int64,sweep::Int64,n::Int64,j_1::Float64,j_2::Float64,h_max::Float64,omega::Float64,two_nn::Bool,hs::Array{Float64},pbc::Bool,psi0::ITensors.MPS,sites::Vector{Index{Int64}})

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


function dmrg_replica(hamiltonian::ITensors.MPO,sweep::Int64,sites::Vector{Index{Int64}},nreplica::Int64,linkdims::Int64,init_bonddim::Int64,set_noise::Bool)
    """ Dmrg operation for different initial radom states"""
    #fix the sweeps
    sweeps = Sweeps(sweep)
    setmaxdim!(sweeps,10,20,40,50,50,60,linkdims)
    setcutoff!(sweeps, 1E-10)
    eng_min::Float64=100000.
    psi_min=nothing #initialize the state that will have minimum energy
    # we could parallelize this part but 
    # i think is difficult
    for i=1:nreplica
        psi0=randomMPS(sites,init_bonddim) #fix the linkdim to 10
        if set_noise
            noise!(sweeps,1E-05,1E-06,1E-06,1E-06,1E-07,1E-08,0)
        end
        energy, psi = dmrg(hamiltonian,psi0, sweeps,outputlevel=1)
        if energy<eng_min
            eng_min=energy
            psi_min=psi
            print("energy min=$eng_min , energy=$energy \n")
        end
    end
    print(inner(psi_min',hamiltonian,psi_min))
    return eng_min,psi_min
end 

function initialize_hamiltonian(j_1::Float64,j_2::Float64,omega::Float64,hs::Array{Float64},two_nn::Bool,pbc::Bool,sites::Vector{Index{Int64}})
    #initialize the hamiltonian
    ham_0=OpSum()
    n=length(hs)
    
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

    # external longitudinal field
    for j=1:n
        ham_ext+=2*omega,"Sx",j # external random field
    end

    hamiltonian=ham_0+ham_ext

    # initialize the hamiltonian
    # as matrix product operator
    h=MPO(hamiltonian,sites) #define the hamiltonian

    return h
end 

function dmrg_nn_ising_check_h_k_map(linkdims::Int64,sweep::Int64,n::Int64,j_1::Float64,j_2::Float64,omega::Float64,h_max::Float64,two_nn::Bool,pbc::Bool,hs::Array{Float64},nreplica::Int64,init_bonddim::Int64,set_noise::Bool,psi0::ITensors.MPS,sites::Vector{Index{Int64}})

    
    #fix the sites
    #sites=siteinds("S=1/2",n) 
    #fix the representation
    #define the universal part of the hamiltonian
    h=initialize_hamiltonian(j_1,j_2,omega,hs,two_nn,pbc,sites)
    



    # energy values
    #energy, psi = dmrg(h,psi0, sweeps,outputlevel=1)
    #energy,psi=dmrg_replica(h,sweep,sites,nreplica,linkdims,init_bonddim,set_noise)


    sweeps = Sweeps(sweep)
    setmaxdim!(sweeps,10,20,40,50,50,60,linkdims)
    setcutoff!(sweeps, 1E-10)
    if set_noise
        noise!(sweeps,1E-05,1E-06,1E-06,1E-06,1E-07,1E-08,0)
    end
    energy, psi = dmrg(hamiltonian,psi0, sweeps,outputlevel=1)
    
    #compute the transverse magnetization and the density functional per site 
    z=2*expect(psi,"Sz")
    # correlation
    zz=4*correlation_matrix(psi,"Sz","Sz")
    return z,zz
end

 

function dmrg_nn_ising_input_output_map(linkdims::Int64,sweep::Int64,n::Int64,j_1::Float64,j_2::Float64,omega::Float64,h_max::Float64,two_nn::Bool,pbc::Bool,hs::Array{Float64},nreplica::Int64,init_bonddim::Int64,set_noise::Bool)

    #fix the sites
    sites=siteinds("S=1/2",n) 
    #fix the representation
    #define the universal part of the hamiltonian
    h=initialize_hamiltonian(j_1,j_2,omega,hs,two_nn,pbc,sites)
    



    # energy values
    #energy, psi = dmrg(h,psi0, sweeps,outputlevel=1)
    energy,psi=dmrg_replica(h,sweep,sites,nreplica,linkdims,init_bonddim,set_noise)

        #compute the transverse magnetization and the density functional per site 
    z=2*expect(psi,"Sz")
    
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
        dens_f=j_1*(x_1nn+x_2nn)
    else
        dens_f=j_2*x_1nn
    end

    #compute the transverse magnetization and the density functional per site 
    # correlation
    zxx=zeros((n,n))
    psi_dag=dag(psi)
    
    for i=1:n
        for r=1:n-2
            f_op_1=OpSum()
            f_op_2=OpSum()
            f_op_1+=j_1,"Sz",i,"Sx",r,"Sx",r+1
            f_op_2+=j_1,"Sz",i,"Sx",r,"Sx",r+2
            f_op_1=MPO(f_op_1,sites)
            f_op_2=MPO(f_op_2,sites)
            zxx[i,r]=8*inner(psi',f_op_1,psi)
            if two_nn
                zxx[i,r]=zxx[i,r]+8*inner(psi',f_op_2,psi)
            end
        end
        if pbc
            f_op_1=OpSum()
            f_op_2=OpSum()
            f_op_1+=j_1,"Sz",i,"Sx",n-1,"Sx",n
            f_op_2+=j_1,"Sz",i,"Sx",n-1,"Sx",1
            f_op_1=MPO(f_op_1,sites)
            f_op_2=MPO(f_op_2,sites)
            zxx[i,n-1]=8*inner(psi',f_op_1,psi)
            if two_nn
                zxx[i,n-1]=zxx[i,n-1]+8*inner(psi',f_op_2,psi)
            end
            f_op_1=OpSum()
            f_op_2=OpSum()
            f_op_1+=j_1,"Sz",i,"Sx",n,"Sx",1
            f_op_2+=j_1,"Sz",i,"Sx",n,"Sx",2
            f_op_1=MPO(f_op_1,sites)
            f_op_2=MPO(f_op_2,sites)
            zxx[i,n]=8*inner(psi',f_op_1,psi)
            if two_nn 
                zxx[i,n]=zxx[i,n]+8*inner(psi',f_op_2,psi)
            end
        end
    end
    
    return zxx,z,dens_f
end


function dmrg_nn_ising_composable(linkdims::Int64,sweep::Int64,n::Int64,j_1::Float64,j_2::Float64,omega::Float64,h_max::Float64,two_nn::Bool,hs::Array{Float64},pbc::Bool,nreplica::Int64,init_bonddim::Int64,set_noise::Bool,psi0::ITensors.MPS,sites::Vector{Index{Int64}})

    #fix the sites
    #sites=siteinds("S=1/2",n) 
    #fix the representation
    #define the universal part of the hamiltonian
    h=initialize_hamiltonian(j_1,j_2,omega,hs,two_nn,pbc,sites)
    



    # energy values
    #energy, psi = dmrg(h,psi0, sweeps,outputlevel=1)
    #energy,psi=dmrg_replica(h,sweep,sites,nreplica,linkdims,init_bonddim,set_noise)

    sweeps = Sweeps(sweep)
    setmaxdim!(sweeps,10,20,40,50,50,60,linkdims)
    setcutoff!(sweeps, 1E-10)
    if set_noise
        noise!(sweeps,1E-05,1E-06,1E-06,1E-06,1E-07,1E-08,0)
    end
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