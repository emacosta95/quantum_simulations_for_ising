#%%
import numpy as np
import matplotlib.pyplot as plt

from src.utils_sparse_diagonalization import transverse_ising_2nn_sparse_simulator


data=np.load("data/temp/data.npz")

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

file_name,es,hs,ms,fs_dens,fs=transverse_ising_2nn_sparse_simulator(l=16,h_max=2.71,hs=pot.reshape(1,-1),n_dataset=1,j1=-1,j2=-1,pbc=True,z_2=False,file_name='nothing')

# %%
plt.plot(dens_f)
plt.plot(fs_dens[0])
plt.show()
# %%
plt.plot(z)
plt.plot(ms[0])
plt.show()
# %%
print(f,fs)
# %%
print(es,e/16)
# %%
