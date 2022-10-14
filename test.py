#%%
import matplotlib.pyplot as plt
import numpy as np

data = np.load("data/dataset_dmrg/l_64_h_2.71_ndata_10.npz")

v = data["potential"]
z = data["density"]
e = data["energy"]
f = data["F"]


print(v.shape)
#%%
for i in range(10):
    plt.plot(v[i])
    plt.plot(z[i], label=f"e={e[i]:.2f}")
    plt.legend()
    plt.show()


# %%
