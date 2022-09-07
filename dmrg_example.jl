#%%
using ITensors
using Distributions
using LinearAlgebra
using Random
using Plots
using NPZ
let
    #parameters
    seed=42
    linkdims=10
    n=16
    j_coupling=1.
    h_max=2.71

    Random.seed!(seed)
    sites=siteinds("S=1",n) #fix the basis representation
    hamiltonian=OpSum()
    potential = Vector{Float64}()
    for i=1:n-2
        hamiltonian+=-j_coupling,"Sx",i,"Sx",i+1 #1 nearest neighbours
        hamiltonian+=-j_coupling,"Sx",i,"Sx",i+2 #2 nearest neighbours
        h_i=rand(Uniform(0,h_max))
        push!(potential,h_i)
        hamiltonian+=h_i,"Sz",i # external random field
    end
    hamiltonian+=-j_coupling,"Sx",n-1,"Sx",n #1 nearest neighbours
    hamiltonian+=-j_coupling,"Sx",n-1,"Sx",1 #2 nearest neighbours
    hamiltonian+=-j_coupling,"Sx",n,"Sx",1 #1 nearest neighbours
    hamiltonian+=-j_coupling,"Sx",n,"Sx",2 #2 nearest neighbours
    h_i=rand(Uniform(0,h_max))
    push!(potential,h_i)
    hamiltonian+=h_i,"Sz",n-1 # external random field
    h_i=rand(Uniform(0,h_max))
    push!(potential,h_i)
    hamiltonian+=h_i,"Sz",n # external random field
    
    
    print(length(hamiltonian))
    
    h=MPO(hamiltonian,sites) #define the hamiltonian
    psi0=randomMPS(sites,linkdims) #initialize the product state

    
    #fix the sweeps
    sweeps = Sweeps(5)
    setmaxdim!(sweeps, 10,20,100,100,200)
    setcutoff!(sweeps, 1E-10)

    # energy values
    energy, psi = dmrg(h,psi0, sweeps)

    #compute the transverse magnetization and the density functional per site 
    z=expect(psi,"Sz")
    xx=correlation_matrix(psi,"Sx","Sx")
    x_1nn=Vector{Float64}()
    x_2nn=Vector{Float64}()   
    for i=1:n
        if i==n 
            push!(x_1nn,xx[i,1])
            push!(x_2nn,xx[i,2])
        elseif i==n-1
            push!(x_1nn,xx[i,i+1])
            push!(x_2nn,xx[i,1])
        else
            push!(x_1nn,xx[i,i+1])
            push!(x_2nn,xx[i,i+2])
        end
    end
    dens_f=-1*x_1nn-x_2nn
    f=energy/n-dot(potential,z)/n

    print(length(f))

    npzwrite("data/temp/data.npz", Dict("density" => z, "energy" => energy, "F" => f,"density_F"=> dens_f,"potential"=>potential))

    # for i=1:n
    #     xx_1nn=expect(psi,"Sx",i,"Sx",(i+1)%n)
    #     xx_2nn=expect(psi,"Sx",i,"Sx",(i+1)%n)
    # end

    # dens_f=xx_1nn + xx_2n

    # # plot the results
    # plot(z)
    # plot(dens_f)
end