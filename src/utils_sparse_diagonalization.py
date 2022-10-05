# import the sparse eigensolver
from typing import Tuple,List,Optional
from scipy.sparse.linalg import eigsh
import numpy as np
from scipy.sparse import csr_matrix
import torch
import matplotlib.pyplot as plt
import quspin
from quspin.operators import hamiltonian,quantum_operator,quantum_LinearOperator # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import argparse
from tqdm import trange

def density_of_functional_pbc(psi:np.array,l:int,basis:quspin.basis,j_1:float,j_2:float):
    m={}
    exp_m=[]
    for i in range(l):
        coupling=[[j_1,i,(i+1)%l],[j_2,i,(i+2)%l]]
        op=['xx',coupling]
        static=[op]
        dynamic=[]
        m=quantum_LinearOperator(static,basis=basis,dtype=np.float64,check_symm=False,check_herm=False,check_pcon=False)
        exp_m.append(m.expt_value(psi))
    return exp_m

def compute_magnetization(psi:np.array,l:int,basis:quspin.basis,direction:str):
    #define the connection
    m={}
    exp_m=[]
    for i in range(l):
        coupling=[[1,i]]
        op=[direction,coupling]
        static=[op]
        dynamic=[]
        m=quantum_LinearOperator(static,basis=basis,dtype=np.float64,check_symm=False,check_herm=False,check_pcon=False)
        exp_m.append(m.expt_value(psi))
    return exp_m


def transverse_ising_sparse_simulator(h_max:int,hs:np.ndarray,n_dataset:int,l:int,j1:float,j2:float,pbc:bool,z_2:bool,file_name:str,check_2nn:bool)->Tuple[str,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:

    #file_name information
    text_z2=''
    text_field=f'_{h_max}_h'

    hs=hs
    #the basis of the representation
    basis=spin_basis_1d(l)

    # the coupling terms
    if pbc:
        j_1nn=[[j1,i,(i+1)%l] for i in range(l)] #pbc
        if check_2nn:
            j_2nn=[[j2,i,(i+2)%l] for i in range(l)] #pbc
    else:
        j_1nn=[[j1,i,(i+1)%l] for i in range(l)] #pbc
        if check_2nn:
            j_2nn=[[j2,i,(i+2)%l] for i in range(l)] #pbc


    for i in trange(n_dataset):
        h=[[hs[i,k],k] for k in range(l)] #external field
        if check_2nn:    
            static=[['xx',j_1nn],['xx',j_2nn],['z',h]]
        else:
            static=[['xx',j_1nn],['z',h]]
        dynamic=[]
        ham=hamiltonian(static,dynamic,basis=basis,dtype=np.float64,check_symm=False,check_herm=False,check_pcon=False)
        e,psi_0=ham.eigsh(k=1)
        z=compute_magnetization(psi_0,l=l,basis=basis,direction='z')
        x=compute_magnetization(psi_0,l=l,basis=basis,direction='x')
        z=np.asarray(z)
        x=np.asarray(x)
        f_dens=density_of_functional_pbc(psi_0,l=l,basis=basis,j_1=j1,j_2=j2)
        f_dens=np.asarray(f_dens)
        f=e[0]/l - np.average(z*hs[i])
        if i==0:
            fs=f
            zs=z.reshape(1,-1)
            fs_dens=f_dens.reshape(1,-1)
            es=e[0]/l
            xs=x.reshape(1,-1)
        else:
            fs=np.append(fs,f)
            zs=np.append(zs,z.reshape(1,-1),axis=0)
            xs=np.append(xs,x.reshape(1,-1),axis=0)
            fs_dens=np.append(fs_dens,f_dens.reshape(1,-1),axis=0)
            es=np.append(es,e[0]/l)
    if z_2:
        text_z2='_augmentation'
        fs=np.append(fs,fs,axis=0)
        fs_dens=np.append(fs_dens,fs_dens,axis=0)
        zs=np.append(zs,-1*zs,axis=0)
        xs=np.append(xs,xs,axis=0)

    dir='data/dataset_1nn'
    if check_2nn:
        dir='data/dataset_2nn'

    file_name=dir+file_name+text_z2+f'_{l}_l_'+text_field+f'_{xs.shape[0]}_n'

    return file_name,es,hs,zs,fs_dens,fs,xs