import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags_array
from scipy import sparse


potFunc_harmonic = lambda x: x**2
potFunc_transmon_test = lambda x: x**2 + np.sin(x)**2
potFunc_transmon = lambda phi: - np.cos(phi)

def get_potfunc_fluxonium(phie, EJ, EL):
    return lambda phi: EJ*np.cos(phi-phie) - 0.5*EL*phi**2

def get_diffop(xmax, fid):
    return get_diffop_sparse(xmax, fid).toarray()
    

def get_diffop_sparse(xmax, fid):
    vecLen = int(2*xmax/fid)
    diffop = np.zeros((vecLen, vecLen), dtype=np.complex128)
    #place entries
    diag1 = np.zeros(vecLen-1, dtype=np.complex128)#have decided diag1 is upper
    diag2 = np.zeros(vecLen, dtype=np.complex128)
    diag3 = np.zeros(vecLen-1, dtype=np.complex128)
    for i in range(vecLen-1):
        diag1[i] += 0.5
        diag3[i] += -0.5
    diffop = diags_array([diag1, diag2, diag3], offsets=[1, 0, -1], shape=(vecLen, vecLen))
    diffop = diffop/fid

    return diffop

def get_diffop_new(xmax,dim):
    vecLen = dim
    diffop = np.zeros((vecLen, vecLen), dtype=np.complex128)
    #place entries
    diag1 = np.zeros(vecLen-1, dtype=np.complex128)#have decided diag1 is upper
    diag2 = np.zeros(vecLen, dtype=np.complex128)
    diag3 = np.zeros(vecLen-1, dtype=np.complex128)
    for i in range(vecLen-1):
        diag1[i] += 0.5
        diag3[i] += -0.5
    diffop = diags_array([diag1, diag2, diag3], offsets=[1, 0, -1], shape=(vecLen, vecLen))
    diffop = diffop/((2*xmax/(dim-1)))

    return diffop
        
def get_ddop_old(xmax,fid):
    return get_ddop_sparse(xmax,fid).toarray()

def get_ddop_new(xmax,dim):
    diag1 = np.zeros(dim-1, dtype=np.complex128)
    diag2 = np.zeros(dim, dtype=np.complex128)
    diag3 = np.zeros(dim-1, dtype=np.complex128)
    for i in range(dim):
        diag2[i] -= 2
        if i < dim-1:
            diag1[i] += 1
            diag3[i] += 1
    dd = diags_array([diag1, diag2, diag3], offsets=[1, 0, -1], shape=(dim, dim))
    dd = dd/((2*xmax/(dim-1))**2)
    return dd

def get_ddop_sparse(xmax,fid):
    vecLen = int(2*xmax/fid)
    #place entries
    diag1 = np.zeros(vecLen-1, dtype=np.complex128)
    diag2 = np.zeros(vecLen, dtype=np.complex128)
    diag3 = np.zeros(vecLen-1, dtype=np.complex128)
    for i in range(vecLen):
        diag2[i] -= 2
        if i < vecLen-1:
            diag1[i] += 1
            diag3[i] += 1
    dd = diags_array([diag1, diag2, diag3], offsets=[1, 0, -1], shape=(vecLen, vecLen))
    dd = dd/(fid**2)
    return dd

def get_potop(xmax, fid, potFunc):
    vecLen = int(2*xmax/fid)
    potop = np.zeros((vecLen, vecLen), dtype=np.complex128)
    for i in range(vecLen):
        potop[i, i] = potFunc(-xmax+i*fid)
    return potop

def get_potop_sparse(xmax, fid, potFunc):
    vecLen = int(2*xmax/fid)
    diag = np.zeros(vecLen, dtype=np.complex128)
    for i in range(vecLen):
        diag[i] = potFunc(-xmax+i*fid)
    potop = diags_array([diag], offsets=[0], shape=(vecLen, vecLen))
    return potop

def get_hamiltonian_sparse(xmax, fid, potFunc):
    dd = get_ddop(xmax, fid)
    potop = get_potop(xmax, fid, potFunc)
    hamiltonian = dd + potop
    return hamiltonian

def get_hamiltonian_transmon(xmax, fid, potFunc):
    nn = get_ddop(xmax, fid)
    potop = get_potop(xmax, fid, potFunc)
    hamiltonian = -4*Ec*nn + potop
    return hamiltonian



def get_n_op(dim,xmax):
    n = np.zeros((dim,dim),dtype=np.complex128)
    for i in range(dim):
        x = -xmax + i*(2*xmax/(dim-1))
        n[i,i] = x
    return n

def get_phi_op(dim,xmax):
    phi = -1j*get_diffop_new(xmax,dim)
    return phi

def get_nsquared_op(dim,xmax):
    nn = np.zeros((dim,dim),dtype=np.complex128)
    for i in range(dim):
        x = -xmax + i*(2*xmax/(dim-1))
        nn[i,i] = x**2
    #nn = nn/(xmax**2)
    #print middle value
    print(nn[int(dim/2),int(dim/2)])
    return nn

def get_phisquared_op(dim,xmax):
    phi = -get_ddop_new(xmax,dim)
    return phi

def get_fluxonium_hammilton_op(dim,xmax,EC,EL,EJ,phi_dc):
    #notice, phi and n are conjugated as compared to the paper, so that the cosine expansion is simpler
    a_squared = get_nsquared_op(dim,xmax)
    b_squared = get_phisquared_op(dim,xmax)
    n = np.linspace(-xmax,xmax,dim)
    #extra term
    extra_term = lambda n: np.cos(n-phi_dc)
    extra_term = np.vectorize(extra_term)
    extra_term = extra_term(n)
    extra_term = np.diag(extra_term)
    #hamiltonian
    ham = 4*EC*a_squared + 0.5*EL*b_squared - EJ*extra_term#the last plus sign is weird
    return ham
    