# %%
import matplotlib.pyplot as plt
import numpy as np

# data
hs = [0.1, 1.2, 2.26, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.8, 10.9, 12.0]

x = {}
z = {}
corr = {}

for h in hs:
    data = np.load(f"data/dmrg_2nn/test_unet_periodic_2nn_l_128_h_{h}_ndata_100.npz")
    x[h] = data["magnetization_x"]
    z[h] = data["density"]
    corr[h] = data["correlation"]

# %% compute the cumulant U4 in log
u4 = []
m2 = []
for h in hs:

    m2.append(np.average(np.average(np.abs(x[h]), axis=-1) ** 2))
    lnm4 = np.log(np.average(x[h] ** 4, axis=-1))
    lnm2 = np.log(np.average(x[h] ** 2, axis=-1))
    lambd = lnm4 - 2 * lnm2
    y = 0.5 * (3 - np.exp(lambd))
    u4.append(np.average(y))

# for h in hs:
#     y=np.average(x[h],axis=0)
#     y = 0.5*(3-(np.average((y)**4, axis=-1) /
#              (np.average((y)**2, axis=-1)**2)))
#     u4.append(y)

# %%
plt.plot(hs, u4)
plt.show()
plt.plot(hs, m2)
plt.show()
# %% Case with l=64
hs2 = [0.1, 1.2, 2.3, 3.4, 4.4, 5.5, 6.6, 7.7, 8.8, 9.8, 10.9, 12.0]

x2 = {}
z2 = {}
corr2 = {}
for h in hs2:
    data = np.load(f"data/dmrg_2nn/test_unet_periodic_2nn_l_64_h_{h}_ndata_100.npz")
    x2[h] = data["magnetization_x"]
    z2[h] = data["density"]
    corr2[h] = data["correlation"]

# %% compute the cumulant U4 in log
u42 = []
m22 = []
for h in hs2:

    m22.append(np.average(np.average(np.abs(x2[h]), axis=-1) ** 2))
    lnm4 = np.log(np.average(x2[h] ** 4, axis=-1))
    lnm2 = np.log(np.average(x2[h] ** 2, axis=-1))
    lambd = lnm4 - 2 * lnm2
    y = 0.5 * (3 - np.exp(lambd))
    u42.append(np.average(y))

# for h in hs:
#     y=np.average(x[h],axis=0)
#     y = 0.5*(3-(np.average((y)**4, axis=-1) /
#              (np.average((y)**2, axis=-1)**2)))
#     u4.append(y)

# %%


# %% Analysis of the correlation functions
