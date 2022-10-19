# %%
import matplotlib.pyplot as plt
import numpy as np

# data
hs = [
    0.10,
    0.36,
    0.61,
    0.87,
    1.13,
    1.39,
    1.65,
    2.16,
    2.42,
    2.67,
    2.94,
    3.19,
    3.45,
    3.71,
    3.97,
    4.23,
    4.48,
    4.74,
    5.6,
    6.7,
    7.8,
    8.9,
    10,
]

x = {}
z = {}

for h in hs:
    data = np.load(f"data/dmrg_2nn/test_unet_periodic_2nn_l_32_h_{h}_ndata_100.npz")
    x[h] = data["magnetization_x"]
    z[h] = data["density"]

# %% compute the cumulant U4 in log
u4 = []
m2 = []
for h in hs:

    m2.append(np.average(x[h] ** 2))
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

    m22.append(np.average(x2[h] ** 2))
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
plt.plot(hs2, u42)
plt.show()
plt.plot(hs2, m22)
plt.plot(hs, m2)
plt.axvline(2 * np.e)
plt.show()

# %% Analysis of the correlation functions
