import os


import numpy as np
from scipy.sparse import diags_array
from numpy import cos,sin
from tqdm import tqdm
import sympy as sp
from copy import copy
import matplotlib._pylab_helpers as pylab_helpers
def is_figure_active():
    return len(pylab_helpers.Gcf.figs) > 0
from time import time
from IPython.display import display as display_
from IPython.display import Math
import qutip as qt
import scipy as scpy

config = {}
with open("config.txt", "r") as f:
    for line in f.readlines():
        line = line.strip()
        config[line.split(":")[0]] = eval(line.split(":")[1])

H0 = 0
n_opp = 0
phi_opp = 0

def display(expr):
    display_(Math(sp.latex(expr)))

def get_nsquared_op(dim,xmax):
    nn = np.zeros((dim,dim),dtype=np.complex128)
    n = np.linspace(-xmax,xmax,dim)
    ns = n**2
    for i in range(dim):
        nn[i,i] = ns[i]
    return nn

def get_n_op(dim,xmax):
    nn = np.zeros((dim,dim),dtype=np.complex128)
    n = np.linspace(-xmax,xmax,dim)
    for i in range(dim):
        nn[i,i] = n[i]
    return nn


def get_ndiff_op(binwidth,n=1):

    dt = sp.Symbol('dt')
    dx = sp.Symbol('dx')
    t = sp.Symbol('t')
    x = sp.Symbol('x')
    g = sp.Function('g')(x)
    m = sp.Symbol('m')
    diff_expr_base = (g.subs(x,x+dx)-g)/dt
    if n == 0: return np.array([1])
    diff_expr = copy(diff_expr_base)
        
    for i in range(n-1):
        gs = [g.subs(x,x+i*dx) for i in range(n)]

        dummies = []
        for i,g_ in enumerate(gs):
            g_dummy1 = sp.Function(f'g_dummy{i}1')(x).subs(x,x+i*dx)
            g_dummy2 = sp.Function(f'g_dummy{i}2')(x+dx).subs(x,x+i*dx)
            dummies.append([g_,g_dummy1])
            dummies.append([g_.subs(x,x+dx),g_dummy2])
            ex_2_sub = copy(diff_expr_base).replace(g,g_dummy1).replace(g.subs(x,x+dx),g_dummy2)
            diff_expr = diff_expr.replace(g_,ex_2_sub)
        #done, now just replace the dummies
        for dummy in dummies:
            diff_expr = diff_expr.replace(dummy[1],dummy[0])
        #simplify
        diff_expr = sp.simplify(diff_expr)
    
    #shift x's by n//2
    diff_expr = diff_expr.subs(x,x-n//2*dx)

    #now evaluate this as a matrix row element:
    diff_expr = diff_expr.subs(dt,binwidth)
    diff_expr = diff_expr.subs(x,0)
    diff_expr = sp.simplify(diff_expr)
    arr_expr = np.zeros(len(diff_expr.args))
    center_index = len(diff_expr.args)//2
    if len(diff_expr.args)%2 == 0: center_index -= 1
    for i,expr in enumerate(diff_expr.args):
        #cound dx's
        dx_count = 0
        number = 0
        for arg in expr.args:
            if dx in arg.free_symbols:
                for a in arg.args:
                    prefac = 1
                    for s in a.expr_free_symbols:
                        try:
                            int(s)
                            prefac *= s
                        except:
                            pass
                    dx_count = prefac
            elif g.subs(x,0) == arg:
                pass
            else:
                number = arg
        #sub g with 1
        arr_expr[center_index+dx_count] = number
    return np.array(arr_expr)

def get_nphi_op(binwidth,dim,n=1):
    arr_fundamental = ((-1j)**n)*get_ndiff_op(binwidth,n).astype(np.complex128)
    matrix = np.zeros((dim,dim),dtype=np.complex128)
    for i in range(dim):
        offset = len(arr_fundamental)//2
        if len(arr_fundamental)%2 == 0: offset -= 1
        for j,e in enumerate(arr_fundamental):
            if i+j-offset >= 0 and i+j-offset < dim:
                matrix[i,i+j-offset] = e
    if n>=1:
        #plot
        fig,ax = plt.subplots(2,1)
        ax[0].imshow(matrix.real[:10,:10])
        ax[1].imshow(matrix.imag[:10,:10])
        plt.savefig(f"nphi_{n}.png")
        plt.show()
        plt.clf()

    return matrix

def get_nphi_op(dim,xmax,n=1):
    #alterrnate: i think i mistook the definition, and this is actually the "number" oppeartor
    diag = np.linspace(-xmax,xmax,dim)
    diag = diag**n
    return np.diag(diag)

def get_nsquared_op(dim,xmax):
    arr_fundamental = -get_ndiff_op(2*xmax/dim,2)
    matrix = np.zeros((dim,dim),dtype=np.complex128)
    for i in range(dim):
        offset = len(arr_fundamental)//2
        if len(arr_fundamental)%2 == 0: offset -= 1
        missing = []
        for j,e in enumerate(arr_fundamental):
            if i+j-offset >= 0 and i+j-offset < dim:
                matrix[i,i+j-offset] = e
            else:
                missing.append(j)
        if len(missing) > 0:
            if i < dim//2:
                for j in missing:
                    matrix[i,0] += arr_fundamental[j]
            elif i > dim//2:
                for j in missing:
                    matrix[i,dim-1] += arr_fundamental[j]
    return matrix

def get_n_op(dim,xmax):
    arr_fundamental = 1j*get_ndiff_op(2*xmax/dim,1)/2
    matrix = np.zeros((dim,dim),dtype=np.complex128)
    for i in range(dim):
        offset = len(arr_fundamental)//2
        if len(arr_fundamental)%2 == 0: offset -= 1
        for j,e in enumerate(arr_fundamental):
            if i+j-offset >= 0 and i+j-offset < dim:
                matrix[i,i+j-offset] = e
    for i in range(dim):
        if i+1 < dim:
            matrix[i+1,i] = matrix[i,i]
        matrix[i,i]=0#quick fix to make hermitian

    return matrix

def get_phi_opp(dim,xmax):
    return get_nphi_op(dim,xmax,1)
    
    
def get_Hamiltonian(Ec,El,Ej,phi_dc, base_ex=np.pi*4, base_size=1000,expansion_order=10,justPot=False,only=""):
    #if "H.npy" not in os.listdir() or justPot:
    #define the hammiltonian in this base
    binwidth = 2*base_ex/base_size
    #dd_op = get_ndiff_op(binwidth,2)
    phi = sp.Symbol('phi')
    n = sp.Symbol('n')
    H= 0.5*El*(phi)**2 - Ej*sp.cos(phi-phi_dc)
    #eval in linspace
    H_e = [H.subs(phi,phi_val) for phi_val in np.linspace(-base_ex,base_ex,base_size)]
    minval = np.min(H_e)
    H = H - minval
    H += 4*Ec*n**2 
    if only != "":
        H = 0
        if "El" in only:
            H += 0.5*El*(phi)**2
        if "Ej" in only:
            H += - Ej*sp.cos(phi-phi_dc)
        if "Ec" in only:
            H += 4*Ec*n**2
    if justPot:
        #strip n, and return as lambdifed function
        H = H.subs(n,0)
        H = sp.lambdify(phi,H)
        #H=H.subs(phi,0)
        #H= sp.lambdify(n,H)
        return H
    #make a taylor series in phi
    H = sp.series(H,phi,n=expansion_order)
    str_H = str(H).split(" + O(")[0]
    #first term is the constant, so replace by a identity
    first = str_H.split("-")[0]
    ident = np.diag(np.ones(base_size))
    str_H = str_H.replace(first,f"ident*{first}")


    #n2_matrix = get_nsquared_op(base_size,base_ex)
    #phis = [get_nphi_op(binwidth,base_size,n=i) for i in tqdm(range(expansion_order))]
    phis = [get_nphi_op(base_size,base_ex,n=i) for i in tqdm(range(expansion_order))]
    n2_matrix = get_nsquared_op(base_size,base_ex)
    #!validate that this is the correct distinction between n and phi (n is equivalent to x, and phi is equivalent to p)
    #shift the energy
    

    #substitute the n^2 and phi^2 operators
    str_H = str_H.replace("n**2","n2_matrix")
    for i,phi in list(enumerate(phis))[::-1]:
        str_H = str_H.replace(f"phi**{i}",f"phis[{i}]")
    #str_H = str_H.replace("phi","phis[1]")
    #assuming no singular n
    try:
        H = eval(str_H)
    except:
        print("Error in eval:",str_H)
        H = None
    #diag = H.diagonal()
    #H += 
    
    #    np.save("H.npy",H)
    #else:
    #    H = np.load("H.npy")
    return H

def solveEi(hamiltonian, truncation,meta=""):
    #eigvals, eigvecs = np.linalg.eigh(hamiltonian)
    #scipy.sparse.linalg.eigs
    print(f"Solving eigvals for {meta}")
    f1,f2 = f"temp/eigvecs_{meta}.npy", f"temp/eigvals_{meta}.npy"
    if f1.split("/")[1] in os.listdir("temp") and f2.split("/")[1] in os.listdir("temp"):
        print("Found cached eigvals, loading...")
        loading_success = True
        eigvecs, eigvals = [],[]
        try:
            eigvecs, eigvals = np.load(f1), np.load(f2)
        except:
            loading_success = False
        if len(eigvecs) != truncation:
            loading_success = False
        if loading_success:
            if len(eigvecs[:,0]) == len(hamiltonian):
                """basisTransform  = np.zeros((len(eigvecs),len(eigvecs)),dtype=np.complex128)
                for i in range(len(eigvecs[0])):
                    basisTransform[:,i] = eigvecs[:,i]"""
                basisTransform = eigvecs
                return eigvals, eigvecs, basisTransform
        else:
            print("Loading failed, recomputing eigvals")
    #hamiltonian = scpy.sparse.csr_matrix(hamiltonian)
    #eigvals, eigvecs = scpy.sparse.linalg.eigsh(hamiltonian,which='SA',k=truncation)
    #do with numpy instead
    eigvals, eigvecs = np.linalg.eigh(hamiltonian)
    print(eigvals)
    print(eigvecs)
    """basisTransform = np.zeros((len(eigvecs),len(eigvecs)),dtype=np.complex128)
    for i in range(len(eigvecs[0])):
        basisTransform[:,i] = eigvecs[:,i]"""
    basisTransform = eigvecs
    mask = np.sum(eigvecs.real, axis=0) < 0
    eigvecs[:, mask] *= -1
    #check all the phases
    for i in range(len(eigvecs[0])):
        for x in eigvecs[:,i]:
            if not np.isclose(np.abs(x),0):
                phase = x/np.abs(x)
                eigvecs[:,i] /= phase
                break
    np.save(f1, eigvecs)
    np.save(f2  , eigvals)
    return eigvals, eigvecs, basisTransform

def get_diag_Hamiltonian(Ec,El,Ej,phi_dc, base_ex=np.pi*4, base_size=500,expansion_order=10,truncation=5):
    H0 = get_Hamiltonian(Ec,El,Ej,phi_dc, base_ex, base_size,expansion_order)
    if H0 is None:
        print("Error in Hamiltonian, returning None")
        return None, None, None, None
    #print(H0-H0.T.conj())
    pot = get_Hamiltonian(Ec,El,Ej,phi_dc, base_ex, base_size,expansion_order,justPot=True)
    PHI = np.linspace(-base_ex, base_ex, base_size)#!validate what is the min/max range???
    pot = pot(PHI)
    #pot -= np.min(pot)
    #get diagonalization matrix
    eigvals, eigvecs, basisTransform = solveEi(H0,truncation,meta=f"{Ec},{El},{Ej},{phi_dc},{base_ex},{base_size},{expansion_order}")
    #sanity
    #Ht  =get_Hamiltonian(Ec,El,Ej,phi_dc, base_ex, base_size,expansion_order,only="ElEj")
    #print(np.sum(np.dot(np.dot(eigvecs[:,0].T.conj(),Ht),eigvecs[:,0])))

    #plot eigenvecs
    
    
    #plt.plot(X, eigvecs[:,0].real, color='red')
    #make cmap for differentiating the plots
    
    #phi = get_nphi_op(2*base_ex/base_size,base_size)
    plt.plot(PHI, pot.real)
    cmap = plt.get_cmap('hsv')
    for i in range(len(eigvals)):
        dE = eigvals[i]
        #eigVec_phi = np.dot(phi, eigvecs[:,i])
        #fourier transform to get in phi space
        #eigVec_phi = np.fft.fft(eigvecs[:,i])
        plt.plot(PHI, eigvecs[:,i].real*5e1+dE, color=cmap(i*1e-1),alpha=np.exp(-i*1e-1),zorder=100-i)
        #plt.plot(PHI, eigvecs[:,i].imag*5e1+dE, color=cmap(i*1e-1),alpha=np.exp(-i*1e-1),zorder=100-i)
        if i > 4: break
    plt.ylim(eigvals[0]-5,eigvals[4]+10)
    #plt.xlim(-np.pi,np.pi)
    plt.savefig("eigenvecs.png")
    fold = os.listdir("temp/eigenvecs")
    values = [int(f.split("_")[-1].split(".")[0]) for f in fold]
    if len(values) == 0: value = 0
    else: value = np.max(values)+1
    plt.savefig(f"temp/eigenvecs/eigenvecs_{value}.png")
    plt.show()
    plt.clf()


    #diagonalized_H = np.dot(np.dot(basisTransform.T, H0), basisTransform)
    diagonalized_H = np.diag(eigvals)
    #print(diagonalized_H-diagonalized_H.T.conj())
    
    #print(diagonalized_H-diagonalized_H.T.conj())
    return diagonalized_H, eigvecs, basisTransform, eigvals
    

from matplotlib import pyplot as plt
from matplotlib import colors

def init_qubit(Ec,El,Ej,phi_dc,c_ops,t_g, base_ex=np.pi*4, base_size=1001,expansion_order=50, truncation=20):
    global H0
    global n_opp
    global phi_opp
    #global c_ops
    #check cops
    for i in range(len(c_ops)):
        c_ops_tmp = np.array(c_ops[i])
        c_ops[i] = np.zeros((truncation,truncation),dtype=np.complex128)
        c_ops[i][:len(c_ops_tmp),:len(c_ops_tmp)] = c_ops_tmp

    H0_diag, eigenvecs, basisTransform, eigenvals = get_diag_Hamiltonian(Ec,El,Ej,phi_dc, base_ex, base_size,expansion_order, truncation=truncation)
    if H0_diag is None:
        print("Error in Hamiltonian, returning None")
        return None, None, None, None, None
    H0_diag = np.diag([eigenvals[i] for i in range(truncation)])-eigenvals[0]*np.eye(truncation)
    n = get_n_op(base_size,base_ex)
    phi = get_nphi_op(base_size,base_ex)
    if config["melems"]:# and not is_figure_active():
        cmap = plt.get_cmap('seismic')
        print("Plotting n and phi")
        fig,ax = plt.subplots(2,2)
        extremum = np.max(np.abs(n))
        ax[0,0].imshow(n.real[:10,:10], vmin=-extremum, vmax=extremum, cmap=cmap)
        ax[0,1].imshow(n.imag[:10,:10], vmin=-extremum, vmax=extremum, cmap=cmap)
        extremum = np.max(np.abs(phi))
        ax[1,0].imshow(phi.real[:10,:10], vmin=-extremum, vmax=extremum, cmap=cmap)
        ax[1,1].imshow(phi.imag[:10,:10], vmin=-extremum, vmax=extremum, cmap=cmap)
        plt.savefig("n_phi_init.png")
        plt.show()
        plt.clf()
        plt.close('all')
    
    #print(n-n.T.conj())
    n = np.dot(np.dot(basisTransform.T.conj(), n), basisTransform)
    #print(n-n.T.conj())
    phi = np.dot(np.dot(basisTransform.T.conj(), phi), basisTransform)

    H0 = H0_diag
    n_opp = n
    phi_opp = phi

    #normalize the operators
    n_opp = n_opp/np.abs(n_opp[0][1])
    if n_opp[0][1].imag > 0:
        n_opp *= -1
    phi_opp = phi_opp/np.abs(phi_opp[0][1])
    if phi_opp[0][1].real < 0:
        phi_opp *= -1

    #truncate
    H0 = H0[:truncation,:truncation]
    n_opp = n_opp[:truncation,:truncation]
    phi_opp = phi_opp[:truncation,:truncation]
    eigenvecs = eigenvecs[:,:truncation]
    basisTransform = basisTransform[:,:truncation]
    eigenvals = eigenvals[:truncation]

    
    
    #plot
    if config["melems"]:# and not is_figure_active():
        print("Plotting n and phi")
        fig,ax = plt.subplots(2,2)
        cmap = plt.get_cmap('seismic')
        extremum = np.max([np.max(np.abs(n_opp)),np.max(np.abs(phi_opp))]) 
        #norm = colors.SymLogNorm(10**1,vmin=-extremum, vmax=extremum) 
        norm = colors.Normalize(vmin=-extremum, vmax=extremum)
        ax[0,0].imshow(n_opp.real[:10,:10], cmap=cmap, norm=norm)
        print(n_opp.real[:10,:10])
        ax[0,0].set_ylabel("n")
        r = ax[0,1].imshow(n_opp.imag[:10,:10], cmap=cmap, norm=norm)
        print(n_opp.imag[:10,:10])
        ax[1,0].imshow(phi_opp.real[:10,:10], cmap=cmap, norm=norm)
        print(phi_opp.real[:10,:10])
        ax[1,0].set_ylabel("phi")
        ax[1,0].set_xlabel("real")
        ax[1,1].imshow(phi_opp.imag[:10,:10], cmap=cmap, norm=norm)
        print(phi_opp.imag[:10,:10])
        ax[1,1].set_xlabel("imaginary")
        #show cbar
        cbar = plt.colorbar(r, ax=ax[0,1], orientation='vertical')
        plt.savefig("n_phi.png")
        plt.show()
        plt.clf()
        plt.close('all')


    
    

    #save
    current_time = int(time())

    for i,c in enumerate(c_ops):
        c = qt.Qobj(np.array(c))
        c_ops[i] = qt.spre(c) - qt.spost(c)

    #print("H0: ",H0)

    return H0, n_opp, phi_opp, c_ops, t_g



#init_qubit(1.3,0.59,5.71,np.pi)