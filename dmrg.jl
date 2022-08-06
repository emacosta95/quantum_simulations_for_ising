#%%
using ITensors
using Distributions
using Random
using Plots
let
    #parameters
    seed=42
    linkdims=10
    n=32
    j_coupling=1.
    h_max=2.71

    Random.seed!(seed)
    sites=siteinds("S=1/2",n) #fix the basis representation
    hamiltonian=OpSum()
    for i=1:n
        if i==n
            hamiltonian+=-j_coupling,"Sx",i,"Sx",1 #1 nearest neighbours
            hamiltonian+=-j_coupling,"Sx",i,"Sx",2 #2 nearest neighbours
        elseif i==n-1
            hamiltonian+=-j_coupling,"Sx",i,"Sx",(i+1) #1 nearest neighbours
            hamiltonian+=-j_coupling,"Sx",i,"Sx",1 #2 nearest neighbours
        else
            hamiltonian+=-j_coupling,"Sx",i,"Sx",(i+1) #1 nearest neighbours
            hamiltonian+=-j_coupling,"Sx",i,"Sx",(i+2) #2 nearest neighbours
        end
        h_i=rand(Uniform(0,h_max))
        hamiltonian+=h_i,"Sz",i # external random field
    end
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
    xx=correlation_function(psi,'Sx','Sx')

    # for i=1:n
    #     xx_1nn=expect(psi,"Sx",i,"Sx",(i+1)%n)
    #     xx_2nn=expect(psi,"Sx",i,"Sx",(i+1)%n)
    # end

    # dens_f=xx_1nn + xx_2n

    # # plot the results
    # plot(z)
    # plot(dens_f)
end