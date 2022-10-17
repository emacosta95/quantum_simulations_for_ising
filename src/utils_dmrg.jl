using ITensors
using Distributions
using LinearAlgebra
using Random
using Plots
using NPZ
using ProgressBars

function dmrg_nn_ising(seed::Int64,linkdims::Int64,sweeps::Int64,n::Int64,j_1::Float64,j_2::Float64,h_max::Float64,ndata::Int64,eps_breaking::Float64,namefile::String,two_nn::Bool)

    #fix the seed
    Random.seed!(seed)

    #fix the representation
    sites=siteinds("S=1/2",n) #fix the basis representation
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
    if two_nn
        ham_0+=4*j_2,"Sx",n-1,"Sx",1 #2 nearest neighbours
    end
    ham_0+=4*j_1,"Sx",n,"Sx",1 #1 nearest neighbours
    if two_nn
        ham_0+=4*j_2,"Sx",n,"Sx",2 #2 nearest neighbours
    end
    #h_i=rand(Uniform(0,h_max))
    #push!(potential,h_i)
    #hamiltonian+=h_i,"Sz",n-1 # external random field
    #h_i=rand(Uniform(0,h_max))
    #push!(potential,h_i)
    #hamiltonian+=h_i,"Sz",n # external random field

    # there is a problem with 
    # the types
    # e_tot = Vector{Float64}()
    # v_tot = [Vector{Float64}() for _ in 1:ndata]
    # f_tot=Vector{Float64}()
    # dens_f_tot= [Vector{Float64}() for _ in 1:ndata]
    # z_tot= [Vector{Float64}() for _ in 1:ndata]
    
    # alternative
    e_tot = zeros(Float64,(ndata))
    v_tot = zeros(Float64,(ndata,n))
    f_tot=zeros(Float64,(ndata))
    dens_f_tot= zeros(Float64,(ndata,n))
    z_tot= zeros(Float64,(ndata,n))
    x_tot=zeros(Float64,(ndata,n))
    #create the dataset
    for i=tqdm(1:ndata)
        #external potential
        ham_ext=OpSum()
        potential = zeros(Float64,n)
        for j=1:n
            h_i=rand(Uniform(0.,h_max))
            potential[j]=h_i
            ham_ext+=2*h_i,"Sz",j # external random field
        end

        for j=1:n
            ham_ext+=2*eps_breaking,"Sx",j # to fix the invariance problem
        end
        # collect the external
        # field
        #push!(v_tot,potential)

        hamiltonian=ham_0+ham_ext

        # initialize the hamiltonian
        # as matrix product operator
        h=MPO(hamiltonian,sites) #define the hamiltonian
        psi0=randomMPS(sites,linkdims) #initialize the product state

    
        #fix the sweeps
        sweeps = Sweeps(sweeps)
        setmaxdim!(sweeps, 10,20,100,100,200)
        setcutoff!(sweeps, 1E-10)

        # energy values
        energy, psi = dmrg(h,psi0, sweeps)

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
        f=energy/n-dot(potential,z)/n
        print(length(f))

        print("type of energy/n ",typeof(potential),"\n")
        # push!(e_tot,energy/n)
        # push!(z_tot,z)
        # push!(dens_f_tot,dens_f)
        # push!(f_tot,f)
        # push!(v_tot,potential)
        
        # alternative method
        e_tot[i]=energy/n
        v_tot[i,:]=potential
        z_tot[i,:]=z
        x_tot[i,:]=x
        dens_f_tot[i,:]=dens_f
        f_tot[i]=f
        
        npzwrite(namefile, Dict("density" => z_tot, "energy" => e_tot, "F" => f_tot,"density_F"=> dens_f_tot,"potential"=>v_tot,"magnetization_x"=>x_tot))

    end

end