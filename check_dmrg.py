#%%
import numpy as np
import matplotlib.pyplot as plt

from src.utils_sparse_diagonalization import transverse_ising_2nn_sparse_simulator


data=np.load("data/dataset_dmrg/l_16_h_2.71_ndata_10.npz")

f=data["F"]
z=data["density"]
dens_f=data["density_F"]
pot=data["potential"]
e=data['energy']

#%%
plt.plot(z)
plt.plot(dens_f)
plt.plot(pot)
plt.show()
# %%

file_name,es,hs,ms,fs_dens,fs=transverse_ising_2nn_sparse_simulator(l=16,h_max=2.71,hs=pot,n_dataset=10,j1=-1,j2=-1,pbc=True,z_2=False,file_name='nothing')

# %%
plt.plot(dens_f[0])
plt.plot(fs_dens[0])
plt.show()

print(es.shape)
print(ms.shape)
# %%
for i in range(10):
    plt.plot(z[i],label=f'e={e[i]:.5f}')
    plt.plot(ms[i],label=f'e={es[i]:.5f}')
    plt.legend()
    plt.show()
    print(np.abs((es[i]-e[i])/e[i]))
# %%
print(f,fs)
# %%
print(es,e/16)
# %%
