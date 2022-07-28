import torch


# exact diagonalization of a spin representation hamiltonian
def local_observable(op:torch.Tensor,site:int,l:int,device:str):
    id=torch.tensor([[1,0],[0,1]]).to(device)
    for i in range(l):

        if i==0 and i!=site:
            a=id
        elif i==0 and i==site:
            a=op
        elif i==site and i!=0:
            a=torch.kron(a,op)
        elif i!=site and i!=0:
            a=torch.kron(a,id)
    return a  

class QuantumSpinSystem():

    def __init__(self,l:int,device:str='cuda'):
        """Create the many-body spin operators for qubit operations

        Args:
            l (int): size of the system
            device (str): device in which the computation is on. Can be either 'cuda' or 'cpu' (standard='cuda')
        """
        #define the pauli operator
        sigma_x=torch.tensor([[0,1],[1,0]]).to(device)
        sigma_y=torch.tensor([[0,-1j],[1j,0]]).to(device)
        sigma_z=torch.tensor([[1,0],[0,-1]]).to(device)

        # initialize the spin site operator
        self.s_x=[]
        self.s_y=[]
        self.s_z=[]
        for i in range(l):
            # consider the local observable
            # for each site 
            a_x=local_observable(sigma_x,l=l,site=i,device=device)
            a_y=local_observable(sigma_y,l=l,site=i,device=device)
            a_z=local_observable(sigma_z,l=l,site=i,device=device)
            self.s_x.append(a_x)
            self.s_y.append(a_y)
            self.s_z.append(a_z)
