# %% using a standard cycle
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.fft import fft, ifft
from scipy.sparse import csr_matrix, kron
from torch import conj
from tqdm import tqdm, trange
from zmq import device

# quantum transverse ising model 1d
torch.manual_seed(42)
torch.set_num_threads(10)


def sparse_local_observable(op: np.array, site: int, l: int):
    id = csr_matrix([[1, 0], [0, 1]])
    for i in range(l):

        if i == 0 and i != site:
            a = id
        elif i == 0 and i == site:
            a = op
        elif i == site and i != 0:
            a = kron(a, op)
        elif i != site and i != 0:
            a = kron(a, id)
    return a


def local_observable(op: torch.Tensor, site: int, l: int, device: str):
    id = torch.tensor([[1, 0], [0, 1]]).to(device)
    for i in range(l):

        if i == 0 and i != site:
            a = id
        elif i == 0 and i == site:
            a = op
        elif i == site and i != 0:
            a = torch.kron(a, op)
        elif i != site and i != 0:
            a = torch.kron(a, id)
    return a


class QuantumSpinSystem:
    def __init__(self, l: int, device: str = "cuda"):
        """Create the many-body spin operators for qubit operations

        Args:
            l (int): size of the system
            device (str): device in which the computation is on. Can be either 'cuda' or 'cpu' (standard='cuda')
        """
        # define the pauli operator
        sigma_x = torch.tensor([[0, 1], [1, 0]]).to(device)
        sigma_y = torch.tensor([[0, -1j], [1j, 0]]).to(device)
        sigma_z = torch.tensor([[1, 0], [0, -1]]).to(device)

        # initialize the spin site operator
        self.s_x = []
        self.s_y = []
        self.s_z = []
        for i in range(l):
            # consider the local observable
            # for each site
            a_x = local_observable(sigma_x, l=l, site=i, device=device)
            a_y = local_observable(sigma_y, l=l, site=i, device=device)
            a_z = local_observable(sigma_z, l=l, site=i, device=device)
            self.s_x.append(a_x)
            self.s_y.append(a_y)
            self.s_z.append(a_z)


class QuantumSpinSystemSparse:
    def __init__(self, l: int):
        """Create the many-body spin operators for qubit operations

        Args:
            l (int): size of the system
            device (str): device in which the computation is on. Can be either 'cuda' or 'cpu' (standard='cuda')
        """
        # define the pauli operator
        sigma_x = csr_matrix([[0, 1], [1, 0]])
        sigma_y = csr_matrix([[0, -1j], [1j, 0]])
        sigma_z = csr_matrix([[1, 0], [0, -1]])

        # initialize the spin site operator
        self.s_x = []
        self.s_y = []
        self.s_z = []
        for i in range(l):
            # consider the local observable
            # for each site
            a_x = sparse_local_observable(sigma_x, l=l, site=i)
            a_y = sparse_local_observable(sigma_y, l=l, site=i)
            a_z = sparse_local_observable(sigma_z, l=l, site=i)
            self.s_x.append(a_x)
            self.s_y.append(a_y)
            self.s_z.append(a_z)


def nambu_diagonalization_ising_model(
    l: int,
    h_max: float,
    j_coupling: float,
    checkpoint: bool,
    train: bool,
    device: str,
    name_file: str,
    pbc: bool,
    n_data: int,
):
    """Diagonalization of the transverse quantum ising model using the Nambu Mapping

    Args:
        l (int): length of the chain
        j_coupling (float): coupling costant of the spin interaction
        hs (np.array): realizations of the magnetic field [batch,l]
        checkpoint (bool): if True, it creates a npz version of the dataset
        train (bool): if True, it labels the file as train, valid otherwise
        device(str): the device used for the computation. Can be either 'cuda' or 'cpu' (standard is 'cpu').
    Returns:
        e,f,m_z (Tuple[np.array]): a triple of energies, H-K functional values and transverse magnetizations for each hs realizations
    """

    n_dataset = n_data
    # uniform means h_ave=0
    hs = np.random.uniform(0, h_max, size=(n_dataset, l))
    hs = torch.tensor(hs, dtype=torch.double, device=device)

    # obc
    j_vec = j_coupling * torch.ones(l, device=device)
    # the 0-th component is null in OBC
    j_vec_l = j_vec.clone()
    if not (pbc):
        j_vec_l[0] = 0
    if pbc:
        j_vec_l[0] = -1 * j_vec_l[0]

    # the l-th component is null in OBC
    j_vec_r = j_vec.clone()
    if not (pbc):
        j_vec_r[-1] = 0
    if pbc:
        j_vec_r[-1] = -1 * j_vec_r[-1]

    # create the nambu matrix

    # create the j matrix in the nearest neighbourhood case
    j_l = torch.einsum("ij,j->ij", torch.eye(l, device=device), j_vec_l)
    j_l = torch.roll(j_l, shifts=-1, dims=1)
    j_r = torch.einsum("ij,j->ij", torch.eye(l, device=device), j_vec_r)
    j_r = torch.roll(j_r, shifts=1, dims=1)
    # the coupling part for a
    j = -0.5 * (j_r + j_l)
    # the coupling part for b
    j_b = -0.5 * (j_r - j_l)
    # the b matrix of the nambu matrix
    b = j_b

    for i in trange(n_dataset):
        # the external field
        h = hs[i]
        h_matrix = torch.einsum("ij,j->ij", torch.eye(l, device=device), h)
        # the a matrix of the nambu matrix
        a = j + h_matrix

        # create the nambu matrix
        h_nambu = torch.zeros((2 * l, 2 * l), device=device)
        h_nambu[:l, :l] = a
        h_nambu[:l, l:] = b
        h_nambu[l:, 0:l] = -1 * torch.conj(b)
        h_nambu[l:, l:] = -1 * torch.conj(a)

        e, u = torch.linalg.eigh(h_nambu)

        # the v coefficients
        v = u.clone()[0:l, l : 2 * l]

        # compute the excitation expectation value
        n = torch.einsum("ln,ln->l", v, torch.conj(v))
        m_z = 1 - 2 * n

        n = torch.einsum("ln,ln->l", v, torch.conj(v))
        m_z = 1 - 2 * n

        e_0 = torch.sum(e[0:l]) / l
        f = e_0 - torch.sum(h * m_z) / l
        if i == 0:
            fs = f.view(1)
            m_zs = m_z.view(1, -1)
            es = e_0.view(1)
        else:
            fs = torch.cat((fs, f.view(1)))
            m_zs = torch.cat((m_zs, m_z.view(1, -1)), dim=0)
            es = torch.cat((es, e_0.view(1)), dim=0)

    if checkpoint:
        if train:
            np.savez(
                f"data/dataset/train_sequential_"
                + name_file
                + f"_{l}_l_{h_max:.1f}_h_{n_dataset}_n.npz",
                energy=es.cpu().numpy(),
                F=fs.cpu().numpy(),
                density=m_zs.cpu().numpy(),
                potential=hs[0 : m_zs.shape[0]].cpu().numpy(),
            )
        else:
            np.savez(
                f"data/dataset/valid_sequential_"
                + name_file
                + f"_{l}_l_{h_max:.1f}_h_{n_dataset}_n.npz",
                energy=es.cpu().numpy(),
                F=fs.cpu().numpy(),
                density=m_zs.cpu().numpy(),
                potential=hs[0 : m_zs.shape[0]].cpu().numpy(),
            )

    return es.cpu().numpy(), fs.cpu().numpy(), m_zs.cpu().numpy()


def parallel_nambu_diagonalization_ising_model(
    nbatch, l: int, j_coupling: float, hs: np.array, device: str, pbc: bool
):
    """Compute the correlation <\sigma_x \sigma_x>(ij) of the transverse quantum ising model using the Nambu Mapping

    Args:
        l (int): length of the chain
        j_coupling (float): coupling costant of the spin interaction
        hs (np.array): realizations of the magnetic field [batch,l]
        checkpoint (bool): if True, it creates a npz version of the dataset
        train (bool): if True, it labels the file as train, valid otherwise
        device(str): the device used for the computation. Can be either 'cuda' or 'cpu' (standard is 'cpu').
    Returns:
        e,f,m_z (Tuple[np.array]): a triple of energies, H-K functional values and transverse magnetizations for each hs realizations
    """

    n_dataset = hs.shape[0]

    batch = int(n_dataset / nbatch)
    # uniform means h_ave=0
    hs = torch.tensor(hs, dtype=torch.double, device=device)

    # obc
    j_vec = j_coupling * torch.ones(l, device=device)
    # the 0-th component is null in OBC
    j_vec_l = j_vec.clone()
    if not (pbc):
        j_vec_l[0] = 0
    if pbc:
        j_vec_l[0] = -1 * j_vec_l[0]

    # the l-th component is null in OBC
    j_vec_r = j_vec.clone()
    if not (pbc):
        j_vec_r[-1] = 0
    if pbc:
        j_vec_r[-1] = -1 * j_vec_r[-1]

    # create the nambu matrix

    # create the j matrix in the nearest neighbourhood case
    j_l = torch.einsum("ij,j->ij", torch.eye(l, device=device), j_vec_l)
    j_l = torch.roll(j_l, shifts=-1, dims=1)
    j_r = torch.einsum("ij,j->ij", torch.eye(l, device=device), j_vec_r)
    j_r = torch.roll(j_r, shifts=1, dims=1)
    # the coupling part for a
    j = -0.5 * (j_r + j_l)
    # the coupling part for b
    j_b = -0.5 * (j_r - j_l)
    # the b matrix of the nambu matrix
    b = j_b

    for i in trange(nbatch):
        # the external field
        h = hs[i * batch : (i + 1) * batch]
        h_matrix = torch.einsum("ij,aj->aij", torch.eye(l, device=device), h)
        # the a matrix of the nambu matrix
        a = j + h_matrix

        # create the nambu matrix
        h_nambu = torch.zeros((batch, 2 * l, 2 * l), device=device)
        h_nambu[:, :l, :l] = a
        h_nambu[:, :l, l:] = b
        h_nambu[:, l:, :l] = -1 * torch.conj(b)
        h_nambu[:, l:, l:] = -1 * torch.conj(a)

        e, w = torch.linalg.eigh(h_nambu)

        # the v coefficients
        v = w.clone()[:, l:, :l]

        u = w.clone()[:, :l, :l]
        # compute the correlation sigma_x sigma_x
        c_vv = torch.einsum("anl,aml->anm", v, torch.conj(v))
        c_uu = torch.einsum("anl,aml->anm", u, torch.conj(u))
        c_vu = torch.einsum("anl,aml->anm", v, torch.conj(u))
        c_uv = torch.einsum("anl,aml->anm", u, torch.conj(v))
        c = c_vv + c_vu - c_uu - c_uv

        s_z = 1 - 2 * torch.einsum("aik,aik->ai", v, torch.conj(v))
        s_z_different = torch.einsum("aik,aik->ai", u, torch.conj(u)) - torch.einsum(
            "aik,aik->ai", v, torch.conj(v)
        )

        density_f = c[:, np.arange(l), (np.arange(l) + 1) % l]
        density_f[:, -1] = -1 * density_f[:, -1]

        e_0 = torch.sum(e[:, 0:l], dim=-1) / l
        f = e_0 - torch.mean(h * s_z, dim=-1)

        if i == 0:
            magn_z = s_z
            magn_z_diff = s_z_different
            e_tot = e_0
            f_tot = f
            tot_density_f = density_f
        else:
            magn_z = np.append(magn_z, s_z, axis=0)
            magn_z_diff = np.append(magn_z_diff, s_z_different, axis=0)
            e_tot = np.append(e_tot, e_0)
            f_tot = np.append(f_tot, f)
            tot_density_f = np.append(tot_density_f, density_f, axis=0)

    return hs, magn_z, magn_z_diff, f_tot, tot_density_f, e_tot


def nambu_correlation_ising_model(
    l: int, j_coupling: float, hs: np.array, device: str, name_file: str, pbc: bool
):
    """Compute the correlation <\sigma_x \sigma_x>(ij) of the transverse quantum ising model using the Nambu Mapping

    Args:
        l (int): length of the chain
        j_coupling (float): coupling costant of the spin interaction
        hs (np.array): realizations of the magnetic field [batch,l]
        checkpoint (bool): if True, it creates a npz version of the dataset
        train (bool): if True, it labels the file as train, valid otherwise
        device(str): the device used for the computation. Can be either 'cuda' or 'cpu' (standard is 'cpu').
    Returns:
        e,f,m_z (Tuple[np.array]): a triple of energies, H-K functional values and transverse magnetizations for each hs realizations
    """

    n_dataset = hs.shape[0]

    # uniform means h_ave=0
    hs = torch.tensor(hs, dtype=torch.double, device=device)

    # obc
    j_vec = j_coupling * torch.ones(l, device=device)
    # the 0-th component is null in OBC
    j_vec_l = j_vec.clone()
    if not (pbc):
        j_vec_l[0] = 0
    if pbc:
        j_vec_l[0] = -1 * j_vec_l[0]

    # the l-th component is null in OBC
    j_vec_r = j_vec.clone()
    if not (pbc):
        j_vec_r[-1] = 0
    if pbc:
        j_vec_r[-1] = -1 * j_vec_r[-1]

    # create the nambu matrix

    # create the j matrix in the nearest neighbourhood case
    j_l = torch.einsum("ij,j->ij", torch.eye(l, device=device), j_vec_l)
    j_l = torch.roll(j_l, shifts=-1, dims=1)
    j_r = torch.einsum("ij,j->ij", torch.eye(l, device=device), j_vec_r)
    j_r = torch.roll(j_r, shifts=1, dims=1)
    # the coupling part for a
    j = -0.5 * (j_r + j_l)
    # the coupling part for b
    j_b = -0.5 * (j_r - j_l)
    # the b matrix of the nambu matrix
    b = j_b

    ss_x = torch.zeros((n_dataset, l, l))
    ss_z = torch.zeros((n_dataset, l, l))
    ss_s_s_z = torch.zeros((n_dataset, l, l))
    for i in trange(n_dataset):
        # the external field
        h = hs[i]
        h_matrix = torch.einsum("ij,j->ij", torch.eye(l, device=device), h)
        # the a matrix of the nambu matrix
        a = j + h_matrix

        # create the nambu matrix
        h_nambu = torch.zeros((2 * l, 2 * l), device=device)
        h_nambu[:l, :l] = a
        h_nambu[:l, l:] = b
        h_nambu[l:, :l] = -1 * torch.conj(b)
        h_nambu[l:, l:] = -1 * torch.conj(a)

        e, w = torch.linalg.eigh(h_nambu)

        # the v coefficients
        v = w.clone()[l:, :l]

        u = w.clone()[:l, :l]
        # compute the correlation sigma_x sigma_x
        c_vv = torch.einsum("nl,ml->nm", v, torch.conj(v))
        c_uu = torch.einsum("nl,ml->nm", u, torch.conj(u))
        c_vu = torch.einsum("nl,ml->nm", v, torch.conj(u))
        c_uv = torch.einsum("nl,ml->nm", u, torch.conj(v))
        c = c_vv + c_vu - c_uu - c_uv

        s_z = 1 - 2 * torch.einsum("ik,ik->i", v, torch.conj(v))
        for k in range(l):
            for r in range(k, l):

                if r != k:
                    ss_z[i, k, r] = ss_z[i, k, r] + (
                        c[k, k] * c[r, r] - c[k, r] * c[r, k]
                    )
                    ss_z[i, r, k] = ss_z[i, r, k] + ss_z[i, k, r]
                    # print(f'k={k},j={j}')
                    ss_x[i, k, r] = ss_x[i, k, r] + torch.linalg.det(
                        c[k:r, k + 1 : r + 1]
                    )
                    ss_x[i, r, k] = ss_x[i, r, k] + ss_x[i, k, r]
                    # print(c[k:j,k+1:j+1])
                    # print(c,k,j)

                else:
                    ss_x[i, k, r] = 1.0
                    ss_z[i, k, r] = 1

    return ss_x, ss_z, s_z


def parallel_nambu_correlation_ising_model(
    nbatch,
    l: int,
    j_coupling: float,
    hs: np.array,
    device: str,
    name_file: str,
    pbc: bool,
):
    """Compute the correlation <\sigma_x \sigma_x>(ij) of the transverse quantum ising model using the Nambu Mapping

    Args:
        l (int): length of the chain
        j_coupling (float): coupling costant of the spin interaction
        hs (np.array): realizations of the magnetic field [batch,l]
        checkpoint (bool): if True, it creates a npz version of the dataset
        train (bool): if True, it labels the file as train, valid otherwise
        device(str): the device used for the computation. Can be either 'cuda' or 'cpu' (standard is 'cpu').
    Returns:
        e,f,m_z (Tuple[np.array]): a triple of energies, H-K functional values and transverse magnetizations for each hs realizations
    """

    n_dataset = hs.shape[0]

    batch = int(n_dataset / nbatch)
    # uniform means h_ave=0
    hs = torch.tensor(hs, dtype=torch.double, device=device)

    # obc
    j_vec = j_coupling * torch.ones(l, device=device)
    # the 0-th component is null in OBC
    j_vec_l = j_vec.clone()
    if not (pbc):
        j_vec_l[0] = 0
    if pbc:
        j_vec_l[0] = -1 * j_vec_l[0]

    # the l-th component is null in OBC
    j_vec_r = j_vec.clone()
    if not (pbc):
        j_vec_r[-1] = 0
    if pbc:
        j_vec_r[-1] = -1 * j_vec_r[-1]

    # create the nambu matrix

    # create the j matrix in the nearest neighbourhood case
    j_l = torch.einsum("ij,j->ij", torch.eye(l, device=device), j_vec_l)
    j_l = torch.roll(j_l, shifts=-1, dims=1)
    j_r = torch.einsum("ij,j->ij", torch.eye(l, device=device), j_vec_r)
    j_r = torch.roll(j_r, shifts=1, dims=1)
    # the coupling part for a
    j = -0.5 * (j_r + j_l)
    # the coupling part for b
    j_b = -0.5 * (j_r - j_l)
    # the b matrix of the nambu matrix
    b = j_b

    for i in trange(nbatch):
        ss_x = torch.zeros((batch, l, l))
        ss_z = torch.zeros((batch, l, l))
        ss_s_s_z = torch.zeros((batch, l, l))
        # the external field
        h = hs[i * batch : (i + 1) * batch]
        h_matrix = torch.einsum("ij,aj->aij", torch.eye(l, device=device), h)
        # the a matrix of the nambu matrix
        a = j + h_matrix

        # create the nambu matrix
        h_nambu = torch.zeros((batch, 2 * l, 2 * l), device=device)
        h_nambu[:, :l, :l] = a
        h_nambu[:, :l, l:] = b
        h_nambu[:, l:, :l] = -1 * torch.conj(b)
        h_nambu[:, l:, l:] = -1 * torch.conj(a)

        e, w = torch.linalg.eigh(h_nambu)

        # the v coefficients
        v = w.clone()[:, l:, :l]

        u = w.clone()[:, :l, :l]
        # compute the correlation sigma_x sigma_x
        c_vv = torch.einsum("anl,aml->anm", v, torch.conj(v))
        c_uu = torch.einsum("anl,aml->anm", u, torch.conj(u))
        c_vu = torch.einsum("anl,aml->anm", v, torch.conj(u))
        c_uv = torch.einsum("anl,aml->anm", u, torch.conj(v))
        c = c_vv + c_vu - c_uu - c_uv

        s_z = 1 - 2 * torch.einsum("aik,aik->ai", v, torch.conj(v))
        for k in range(l):
            for r in range(k, l):
                if r != k:
                    ss_z[:, k, r] = ss_z[:, k, r] + (
                        c[:, k, k] * c[:, r, r] - c[:, k, r] * c[:, r, k]
                    )
                    ss_z[:, r, k] = ss_z[:, r, k] + ss_z[:, k, r]
                    # print(f'k={k},j={j}')
                    ss_x[:, k, r] = ss_x[:, k, r] + torch.linalg.det(
                        c[:, k:r, k + 1 : r + 1]
                    )
                    ss_x[:, r, k] = ss_x[:, r, k] + ss_x[:, k, r]
                    # print(c[k:j,k+1:j+1])
                    # print(c,k,j)
                else:
                    ss_x[:, k, r] = 1.0
                    ss_z[:, k, r] = 1
        if i == 0:
            corr_zz = ss_z
            magn_z = s_z
            corr_xx = ss_x
        else:
            corr_zz = np.append(corr_zz, ss_z, axis=0)
            corr_xx = np.append(corr_xx, ss_x, axis=0)
            magn_z = np.append(magn_z, s_z, axis=0)

    return corr_xx, corr_zz, magn_z


# %%
def nambu_energy_gap_ising_model(
    l: int, h_max: float, j_coupling: float, device: str, pbc: bool, n_data: int
):
    """Diagonalization of the transverse quantum ising model using the Nambu Mapping

    Args:
        l (int): length of the chain
        j_coupling (float): coupling costant of the spin interaction
        hs (np.array): realizations of the magnetic field [batch,l]
        checkpoint (bool): if True, it creates a npz version of the dataset
        train (bool): if True, it labels the file as train, valid otherwise
        device(str): the device used for the computation. Can be either 'cuda' or 'cpu' (standard is 'cpu').
    Returns:
        e,f,m_z (Tuple[np.array]): a triple of energies, H-K functional values and transverse magnetizations for each hs realizations
    """

    n_dataset = n_data
    # uniform means h_ave=0
    hs = np.random.uniform(0, h_max, size=(n_dataset, l))
    hs = torch.tensor(hs, dtype=torch.double, device=device)

    # obc
    j_vec = j_coupling * torch.ones(l, device=device)
    # the 0-th component is null in OBC
    j_vec_l = j_vec.clone()
    if not (pbc):
        j_vec_l[0] = 0
    if pbc:
        j_vec_l[0] = -1 * j_vec_l[0]

    # the l-th component is null in OBC
    j_vec_r = j_vec.clone()
    if not (pbc):
        j_vec_r[-1] = 0
    if pbc:
        j_vec_r[-1] = -1 * j_vec_r[-1]

    # create the nambu matrix

    # create the j matrix in the nearest neighbourhood case
    j_l = torch.einsum("ij,j->ij", torch.eye(l, device=device), j_vec_l)
    j_l = torch.roll(j_l, shifts=-1, dims=1)
    j_r = torch.einsum("ij,j->ij", torch.eye(l, device=device), j_vec_r)
    j_r = torch.roll(j_r, shifts=1, dims=1)
    # the coupling part for a
    j = -0.5 * (j_r + j_l)
    # the coupling part for b
    j_b = -0.5 * (j_r - j_l)
    # the b matrix of the nambu matrix
    b = j_b

    for i in trange(n_dataset):
        # the external field
        h = hs[i]
        h_matrix = torch.einsum("ij,j->ij", torch.eye(l, device=device), h)
        # the a matrix of the nambu matrix
        a = j + h_matrix

        # create the nambu matrix
        h_nambu = torch.zeros((2 * l, 2 * l), device=device)
        h_nambu[:l, :l] = a
        h_nambu[:l, l:] = b
        h_nambu[l:, 0:l] = -1 * torch.conj(b)
        h_nambu[l:, l:] = -1 * torch.conj(a)

        e, u = torch.linalg.eigh(h_nambu)
        e_0 = -2 * e[l] / l
        if i == 0:
            es = e_0.view(1)
        else:
            es = torch.cat((es, -2 * e_0.view(1) / l), dim=0)

    return es.cpu().numpy()


def nambu_from_magnetization_to_energy_density(
    nbatch: int, l: int, j_coupling: float, hs: np.array, device: str, pbc: bool
):
    """Compute the correlation <\sigma_x \sigma_x>(ij) of the transverse quantum ising model using the Nambu Mapping

    Args:
        l (int): length of the chain
        j_coupling (float): coupling costant of the spin interaction
        hs (np.array): realizations of the magnetic field [batch,l]
        checkpoint (bool): if True, it creates a npz version of the dataset
        train (bool): if True, it labels the file as train, valid otherwise
        device(str): the device used for the computation. Can be either 'cuda' or 'cpu' (standard is 'cpu').
    Returns:
        e,f,m_z (Tuple[np.array]): a triple of energies, H-K functional values and transverse magnetizations for each hs realizations
    """

    n_dataset = hs.shape[0]

    # uniform means h_ave=0
    hs = torch.tensor(hs, dtype=torch.double, device=device)

    # obc
    j_vec = j_coupling * torch.ones(l, device=device)
    # the 0-th component is null in OBC
    j_vec_l = j_vec.clone()
    if not (pbc):
        j_vec_l[0] = 0
    if pbc:
        j_vec_l[0] = -1 * j_vec_l[0]

    # the l-th component is null in OBC
    j_vec_r = j_vec.clone()
    if not (pbc):
        j_vec_r[-1] = 0
    if pbc:
        j_vec_r[-1] = -1 * j_vec_r[-1]

    # create the nambu matrix

    # create the j matrix in the nearest neighbourhood case
    j_l = torch.einsum("ij,j->ij", torch.eye(l, device=device), j_vec_l)
    j_l = torch.roll(j_l, shifts=-1, dims=1)
    j_r = torch.einsum("ij,j->ij", torch.eye(l, device=device), j_vec_r)
    j_r = torch.roll(j_r, shifts=1, dims=1)
    # the coupling part for a
    j = -0.5 * (j_r + j_l)
    # the coupling part for b
    j_b = -0.5 * (j_r - j_l)
    # the b matrix of the nambu matrix
    b = j_b

    ss_x = torch.zeros((n_dataset, l, l))
    ss_z = torch.zeros((n_dataset, l, l))
    ss_s_s_z = torch.zeros((n_dataset, l, l))

    batch = int(n_dataset / nbatch)
    for i in trange(nbatch):
        # the external field
        h = hs[i * batch : (i + 1) * batch]
        h_matrix = torch.einsum("ij,aj->aij", torch.eye(l, device=device), h)
        # the a matrix of the nambu matrix
        a = j[None, :, :] + h_matrix

        # create the nambu matrix
        h_nambu = torch.zeros((batch, 2 * l, 2 * l), device=device)
        h_nambu[:, :l, :l] = a
        h_nambu[:, :l, l:] = b
        h_nambu[:, l:, :l] = -1 * torch.conj(b)
        h_nambu[:, l:, l:] = -1 * torch.conj(a)

        e, w = torch.linalg.eigh(h_nambu)

        # the v coefficients
        v = w.clone()[:, l:, :l]

        u = w.clone()[:, :l, :l]
        # compute the correlation sigma_x sigma_x
        c_vv = torch.einsum("anl,aml->anm", v, torch.conj(v))
        c_uu = torch.einsum("anl,aml->anm", u, torch.conj(u))
        c_vu = torch.einsum("anl,aml->anm", v, torch.conj(u))
        c_uv = torch.einsum("anl,aml->anm", u, torch.conj(v))
        c = c_vv + c_vu - c_uu - c_uv

        s_z = 1 - 2 * torch.einsum("aik,aik->ai", v, torch.conj(v))
        density_f = c[:, np.arange(l), (np.arange(l) + 1) % l]
        density_f[:, -1] = -1 * density_f[:, -1]
        # print(c[0,0,1],density_f[0,0])
        # print(density_f.shape)
        if i == 0:
            tot_sz = s_z
            tot_density_f = density_f
        else:
            tot_sz = np.append(tot_sz, s_z, axis=0)
            tot_density_f = np.append(tot_density_f, density_f, axis=0)

    return tot_sz, tot_density_f
