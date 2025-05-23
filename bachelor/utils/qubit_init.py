import os


import numpy as np
#from scipy.sparse import diags_array
import scipy
from numpy import cos,sin
from tqdm import tqdm
import sympy as sp
from copy import copy
import matplotlib._pylab_helpers as pylab_helpers
def is_figure_active():
    return len(pylab_helpers.Gcf.figs) > 0
from time import time, sleep
from IPython.display import display as display_
from IPython.display import Math
import qutip as qt
#import qutip_cupy
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
    #return np.diag(diag)
    return scipy.sparse.diags_array([diag], offsets=[0], shape=(dim,dim), dtype=np.complex128)

def get_nsquared_op(dim,xmax,periodic=False):
    arr_fundamental = -get_ndiff_op(2*xmax/dim,2)
    matrix = np.zeros((dim,dim),dtype=np.complex128)
    for i in range(dim):
        offset = len(arr_fundamental)//2
        if len(arr_fundamental)%2 == 0: offset -= 1
        if not periodic:
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
        else:
            for j,e in enumerate(arr_fundamental):
                index = (i+j-offset)%dim
                matrix[i,index] = e
    return matrix

def get_nsquared_op(dim,xmax,periodic=False):#the same but sparse
    arr_fundamental = -get_ndiff_op(2*xmax/dim,2)
    #matrix = np.zeros((dim,dim),dtype=np.complex128)
    diags = [np.full((dim),arr_fundamental[i]) for i in range(len(arr_fundamental))]
    diags[0] = diags[0][:-1]
    diags[-1] = diags[-1][1:]
    matrix = scipy.sparse.diags_array(diags, offsets=[1,0,-1], shape=(dim,dim), dtype=np.complex128)
    return matrix
    """for i in range(dim):
        offset = len(arr_fundamental)//2
        if len(arr_fundamental)%2 == 0: offset -= 1
        if not periodic:
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
        else:
            for j,e in enumerate(arr_fundamental):
                index = (i+j-offset)%dim
                matrix[i,index] = e
    return matrix"""

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

def get_n_op(dim,xmax):
    #the above but sparse
    arr_fundamental = 1j*get_ndiff_op(2*xmax/dim,1)/2
    #matrix = np.zeros((dim,dim),dtype=np.complex128)
    diags = [np.full((dim),arr_fundamental[i]) for i in range(len(arr_fundamental))]
    diags[0] = diags[0][:-1]
    diags[-1] = diags[-1][1:]
    matrix = scipy.sparse.diags_array(diags, offsets=[1,-1], shape=(dim,dim), dtype=np.complex128)

    return matrix


def get_phi_opp(dim,xmax):
    return get_nphi_op(dim,xmax,1)
    
    
def get_Hamiltonian(Ec,El,Ej,phi_dc,omega_01,alpha, base_ex=np.pi*4, base_size=1000,expansion_order=10,justPot=False,only=""):
    #if "H.npy" not in os.listdir() or justPot:
    #define the hammiltonian in this base
    binwidth = 2*base_ex/base_size
    if El == 0: periodic = True
    else: periodic = False
    #dd_op = get_ndiff_op(binwidth,2)
    phi = sp.Symbol('phi')
    n = sp.Symbol('n')
    H= 0.5*El*(phi)**2 - Ej*sp.cos(phi-phi_dc)
    """H= 0.5*El*(phi)**2 - Ej*sp.cos(phi-phi_dc)
    #eval in linspace
    H_e" = [H.subs(phi,phi_val) for phi_val in np.linspace(-base_ex,base_ex,base_size)]"""
    # the above, but without use of scipy
    func = lambda phi_: 0.5*El*((phi_)**2) - Ej*np.cos(phi_-np.full(base_size,phi_dc))
    H_e = func(np.linspace(-base_ex,base_ex,base_size))
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
    #H = sp.series(H,phi,n=expansion_order)
    args = H.args
    const = [a for a in args if phi not in a.free_symbols and n not in a.free_symbols][0]
    H -= const
    str_H = str(H)#.split(" + O(")[0]
    #first term is the constant, so replace by a identity
    #first = str_H.split("-")[0]
    #ident = np.diag(np.ones(base_size))
    ident = scipy.sparse.diags_array([np.ones(base_size)], offsets=[0], shape=(base_size,base_size), dtype=np.complex128)
    #str_H = str_H.replace(first,f"ident*{first}")


    #n2_matrix = get_nsquared_op(base_size,base_ex)
    #phis = [get_nphi_op(binwidth,base_size,n=i) for i in tqdm(range(expansion_order))]
    #phis = [get_nphi_op(base_size,base_ex,n=i) for i in tqdm(range(expansion_order))]
    n2_matrix = get_nsquared_op(base_size,base_ex,periodic=periodic)
    #!validate that this is the correct distinction between n and phi (n is equivalent to x, and phi is equivalent to p)
    #shift the energy
    

    #substitute the n^2 and phi^2 operators
    #phi_matrix = np.diag(np.linspace(-base_ex,base_ex,base_size))
    phi_matrix = scipy.sparse.diags_array([np.linspace(-base_ex,base_ex,base_size)], offsets=[0], shape=(base_size,base_size), dtype=np.complex128)
    #temp
    if False:
        str_H = str_H.replace("n**2","0")
        str_H = str_H.replace("phi**2","0")
    str_H = str_H.replace("n**2","n2_matrix")
    #str_H = str_H.replace("cos(", "ident*np.cos(")
    #eval cos part manually:
    cospart_str = str_H.split("cos(")[1].split(")")[0]
    input = eval(cospart_str.replace("phi","np.linspace(-base_ex,base_ex,base_size)"))
    cospart = scipy.sparse.diags_array([np.cos(input)], offsets=[0], shape=(base_size,base_size), dtype=np.complex128)
    str_H = str_H.replace(f"cos({cospart_str})","cospart")

    """for i,phi in list(enumerate(phis))[::-1]:
        str_H = str_H.replace(f"phi**{i}",f"phis[{i}]")"""
    str_H = str_H.replace("phi","phi_matrix")
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

#import cupy
import cupyx
from cupyx.scipy.sparse.linalg import eigsh
import random
import cupy
import GPUtil
import subprocess

def solveEi(hamiltonian, truncation,meta="",gpu=True):
    #eigvals, eigvecs = np.linalg.eigh(hamiltonian)
    #scipy.sparse.linalg.eigs
    print(f"Solving eigvals for {meta}")
    f1,f2 = f"temp/eigvecs_{meta}.npy", f"temp/eigvals_{meta}.npy"
    #set random seed
    #get random seed from curl
    if hamiltonian.shape[0] <= 3000:
        gpu = False
    if gpu:
        result = subprocess.check_output('curl "http://www.randomnumberapi.com/api/v1.0/random?min=100&max=1000&count=5" ', shell=True, text=True)
        random.seed(result)
    def wait4gpu():
        result = subprocess.check_output('curl "http://www.randomnumberapi.com/api/v1.0/random?min=100&max=1000&count=5" ', shell=True, text=True)
        #rlocal = copy.deepcopy(random)
        random.seed(result)
        rnd = random.uniform(0, 3)
        if gpu:
            print(f"Waiting {rnd} seconds for GPU")
            sleep(rnd)
        gpu_availability = GPUtil.getAvailability(GPUtil.getGPUs(), maxLoad = 0.5, maxMemory = 0.5, includeNan=False, excludeID=[], excludeUUID=[])[:1]
        while np.sum(gpu_availability) == 0:
            sleep(1)
            gpu_availability = GPUtil.getAvailability(GPUtil.getGPUs(), maxLoad = 0.5, maxMemory = 0.5, includeNan=False, excludeID=[], excludeUUID=[])[:1]
        return gpu_availability
    if gpu:
        wait4gpu()
    if f1.split("/")[1] in os.listdir("temp") and f2.split("/")[1] in os.listdir("temp"):
        print("Found cached eigvals, loading...")
        loading_success = True
        eigvecs, eigvals = [],[]
        try:
            eigvecs, eigvals = np.load(f1), np.load(f2)
        except:
            loading_success = False
        eigvecs = np.array(eigvecs)
        eigvals = np.array(eigvals)
        if len(eigvecs.T) != truncation:
            loading_success = False
        if loading_success:
            if len(eigvecs) == hamiltonian.shape[0]:
                """basisTransform  = np.zeros((len(eigvecs),len(eigvecs)),dtype=np.complex128)
                for i in range(len(eigvecs[0])):
                    basisTransform[:,i] = eigvecs[:,i]"""
                basisTransform = eigvecs
                return eigvals, eigvecs, basisTransform
        else:
            print("Loading failed, recomputing eigvals")
            pass

    #eigvals, eigvecs = np.linalg.eigh(hamiltonian)
    #make the hamiltonian sparse
    #H = scpy.sparse.csr_matrix(hamiltonian)
    #we know the diag, and the two off-diags are the only non-sparse elements, extract these, and redefine H
    if type(hamiltonian) == np.ndarray:
        H_d1 = np.diag(hamiltonian,1)
        H_d2 = np.diag(hamiltonian,-1)
        H_d0 = np.diag(hamiltonian)
        H = scipy.sparse.diags([H_d1,H_d2,H_d0], [1,-1,0], format="csr")
    elif gpu:
        H_d1 = hamiltonian.diagonal(1)
        H_d2 = hamiltonian.diagonal(-1)
        H_d0 = hamiltonian.diagonal()
    else:
        H = hamiltonian
    #H = scpy.sparse.diags([H_d1,H_d2,H_d0], [1,-1,0], format="csr")
    #solve
    #eigvals, eigvecs = scpy.sparse.linalg.eigs(H, k=truncation, which='SR', return_eigenvectors=True)
    #the above but with cupy
    if gpu:
        avail = wait4gpu()
        device_id = np.argmax(avail)
        with cupy.cuda.Device(device_id):
            #while gpu utils says the gpu is busy, wait
            #while GPUtil.getGPUs()[device_id].
            print(f'Array created on GPU {device_id}')
            H = cupyx.scipy.sparse.diags([H_d1,H_d2,H_d0], [1,-1,0], format="csr")
            eigvals, eigvecs = eigsh(H, return_eigenvectors=True, k=truncation)
            eigvals_ = cupy.asnumpy(eigvals)
            eigvecs_ = cupy.asnumpy(eigvecs)
            del eigvals, eigvecs, H
            eigvals = eigvals_
            eigvecs = eigvecs_
    else:
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(H, return_eigenvectors=True, which='SR')
        sort = np.argsort(eigvals)
        eigvals = eigvals[sort]
        eigvecs = eigvecs[:,sort]
 
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
    if config["eigenvecs"]:
        #plot
        plt.clf()
        plt.close('all')
        plt.figure(figsize=(10,5))
        cmap = plt.get_cmap('hsv')
        for i in range(len(eigvals)):
            plt.plot(eigvecs[:,i].real, color=cmap(i*1e-1),alpha=np.exp(-i*1e-1),zorder=100-i)
            #plt.plot(eigvecs[:,i].imag, color=cmap(i*1e-1),alpha=np.exp(-i*1e-1),zorder=100-i)
            if i > 4: break
        plt.savefig("eigenvecs.png")
        plt.show()
        plt.clf()
        plt.close('all')

    np.save(f1, eigvecs)
    np.save(f2  , eigvals)
    return eigvals, eigvecs, basisTransform

import matplotlib
import pathlib
def get_diag_Hamiltonian(Ec,El,Ej,phi_dc,omega_01,alpha, base_ex=np.pi*4, base_size=500,expansion_order=10,truncation=5,periodic=False,optim=True, Ec_IC=None, El_IC=None, Ej_IC=None):
    #eigvals_landscape = []
    gradient = [np.inf,np.inf]#what to gain from varying the specific variables
    vary_mode = 0
    if El == 0: periodic = True
    else: periodic = False

    #plot eigenvecs
    def plot_eig(Ec,El,Ej,phi_dc,omega_01,alpha, base_ex, base_size):
        pot = get_Hamiltonian(Ec,El,Ej,phi_dc,omega_01,alpha, base_ex, base_size,expansion_order,justPot=True)
        PHI = np.linspace(-base_ex, base_ex, base_size)#!validate what is the min/max range???
        pot = pot(PHI)
        #get only 1000 points
        indecies = np.linspace(0, base_size-1, 1000).astype(int)
        PHI = PHI[indecies]
        pot = pot[indecies]
        eigv_t = [eigvecs.T[i][indecies] for i in range(len(eigvals))]
    
    
        #plt.plot(X, eigvecs[:,0].real, color='red')
        #make cmap for differentiating the plots
        
        #phi = get_nphi_op(2*base_ex/base_size,base_size)
        if config["eigenvecs"]:
            plt.clf()
            plt.close('all')
            plt.figure(figsize=(10,5))
            cmap = plt.get_cmap('hsv')
            plt.plot(PHI, pot.real)
            
            for i in range(len(eigvals)):
                dE = eigvals[i] - np.min(eigvals)#!temp
                #eigVec_phi = np.dot(phi, eigvecs[:,i])
                #fourier transform to get in phi space
                #eigVec_phi = np.fft.fft(eigvecs[:,i])
                plt.plot(PHI, eigv_t[i].real*5e1+dE, color=cmap(i*1e-1),alpha=np.exp(-i*1e-1),zorder=100-i)
                #plt.plot(PHI, eigvecs[:,i].imag*5e1+dE, color=cmap(i*1e-1),alpha=np.exp(-i*1e-1),zorder=100-i)
                if i > 4: break
            plt.ylim(eigvals[0]-5- np.min(eigvals),eigvals[2]+10- np.min(eigvals))

            plt.savefig("eigenvecs.png")
        fold = os.listdir("temp/eigenvecs")
        values = [int(f.split("_")[-1].split(".")[0]) for f in fold]
        if len(values) == 0: value = 0
        else: value = np.max(values)+1
        plt.savefig(f"temp/eigenvecs/eigenvecs_{value}.png")
        plt.show()
        plt.clf()
        plt.close('all')
    if omega_01 != None and alpha != None and optim:
        params2opt = []
        if Ec == None: params2opt.append("Ec")
        if El == None: params2opt.append("El")
        if Ej == None: params2opt.append("Ej")
        initial = np.ones(len(params2opt))
        idx = 0
        if Ec_IC != None and Ec == None: 
            initial[idx] = Ec_IC
            idx += 1
        if El_IC != None and El == None:
            initial[idx] = El_IC
            idx += 1
        if Ej_IC != None and Ej == None:
            initial[idx] = Ej_IC
            idx += 1

        #initial[-1] = 10
        fixed = []
        if Ec != None: fixed.append(Ec)
        if El != None: fixed.append(El)
        if Ej != None: fixed.append(Ej)
        #optimize
        Els, Ejs, Ecs = [],[],[]
        losses = []
        def task(cord):
            indx_o = 0
            indx_f = 0
            if "Ec" in params2opt:
                Ec_ = cord[indx_o]
                indx_o += 1
            else:
                Ec_ = fixed[indx_f]
                indx_f += 1
            if "El" in params2opt:
                El_ = cord[indx_o]
                indx_o += 1
            else:
                El_ = fixed[indx_f]
                indx_f += 1
            if "Ej" in params2opt:
                Ej_ = cord[indx_o]
                indx_o += 1
            else:
                Ej_ = fixed[indx_f]
                indx_f += 1
            if Ec_ < 0 or El_ < 0 or Ej_ < 0:
                return np.inf
            H0 = get_Hamiltonian(Ec_,El_,Ej_,phi_dc,omega_01,alpha, 6*np.pi, 2000,expansion_order)
            if H0 is None:
                return np.inf
            eigvals, _, _ = solveEi(H0,truncation,meta=f"{Ec_},{El_},{Ej_},{phi_dc},{6*np.pi},{2000},{expansion_order}")
            omega_01_current = eigvals[1]-eigvals[0]
            alpha_current = eigvals[2]-eigvals[1] - omega_01_current
            loss = lambda o,a: np.abs(o-omega_01)**4 + np.abs(a-alpha)**2
            l = loss(omega_01_current,alpha_current)
            Els.append(El_)
            Ejs.append(Ej_)
            Ecs.append(Ec_)
            losses.append(l)
            #plot (just El and Ej)
            if len(losses) > 1:
                plt.clf()
                l_2plt = losses[-20:]
                J_2plt = Ejs[-20:]
                C_2plt = Ecs[-20:]
                L_2plt = Els[-20:]
                plt.scatter(J_2plt,C_2plt,c=l_2plt,norm=matplotlib.colors.LogNorm(),s=50)
                plt.xlabel("Ej")
                plt.ylabel("Ec")
                plt.colorbar()
                plt.savefig("losses.png")
                plt.show()
                plt.clf()
                plt.close('all')

            return l
        #optimize
        from scipy.optimize import minimize
        import pickle
        meta_name = f"qubitoptimm_{Ec},{El},{Ej},{phi_dc},{omega_01},{alpha},{base_ex},{base_size},{expansion_order}"
        print(f"Optimizing {meta_name}")
        if os.path.exists(f"temp/{meta_name}.pickle"):
            print("Found cached optimization, loading...")
            while os.path.exists(f"temp/{meta_name}.dummy"):
                print("Waiting for other process to finish...")
                sleep(1)
            with open(f"temp/{meta_name}.pickle", "rb") as f:
                result = pickle.load(f)
                print("Loaded")
        else:
            pathlib.Path(f"temp/{meta_name}.dummy").touch()
            result = minimize(task, initial, method='Nelder-Mead', options={'disp': True, 'xatol': 1e-2})
            with open(f"temp/{meta_name}.pickle", "wb") as f:
                pickle.dump(result, f)
                print("Saved")
            try:
                os.remove(f"temp/{meta_name}.dummy")
            except FileNotFoundError:
                pass
        r = result.x
        indx = 0
        if "Ec" in params2opt:
            Ec = r[indx]
            indx += 1
        if "El" in params2opt:
            El = r[indx]
            indx += 1
        if "Ej" in params2opt:
            Ej = r[indx]
            indx += 1
        print(f"Optimized: {result.x}")
    def sol(Ec,El,Ej,phi_dc,omega_01,alpha, base_ex, base_size,expansion_order):
        #get Hamiltonian
        H0 = get_Hamiltonian(Ec,El,Ej,phi_dc,omega_01,alpha, base_ex, base_size,expansion_order)
        if H0 is None:
            print("Error in Hamiltonian, returning None")
            return None, None, None, None
        #get diagonalization matrix
        eigvals, eigvecs, basisTransform = solveEi(H0,truncation,meta=f"{Ec},{El},{Ej},{phi_dc},{base_ex},{base_size},{expansion_order}")
        return H0, eigvals, eigvecs, basisTransform
    H0, eigvals, eigvecs, basisTransform = sol(Ec,El,Ej,phi_dc,omega_01,alpha, base_ex, base_size,expansion_order)
    last_eigvals = eigvals
    i = 0
    last_gradient = [np.inf,np.inf]
    while not np.array([gradient[0]<0.0001, gradient[1]<0.0001]).all() and truncation > 2 and optim:
        if vary_mode == 0:
            #vary base size
            base_size += 1000
        elif vary_mode == 1:
            #vary base ex
            base_ex += np.pi/2
        H0, eigvals, eigvecs, basisTransform = sol(Ec,El,Ej,phi_dc,omega_01,alpha, base_ex, base_size,expansion_order)
        #find the gain from this (gradient)
        diff = np.abs(eigvals[:3]-last_eigvals[:3])
        if vary_mode == 0:
            gradient[0] = np.mean(diff)
        elif vary_mode == 1:
            gradient[1] = np.mean(diff)#bias
        last_eigvals = eigvals
        if np.any([last<current for last,current in zip(last_gradient,gradient)]):
            #if this is the case, we expect a numerical error, so instead of continuing, shrink the matrix and end itteration
            if vary_mode == 0:
                base_size -= 1000
            elif vary_mode == 1:
                base_ex -= np.pi/2
            break
        if gradient[0] < gradient[1]:
            vary_mode = 1
        elif gradient[0] > gradient[1] and not periodic:
            vary_mode = 0
        
        print(gradient)
        last_gradient = copy(gradient)
        i+= 1

        plot_eig(Ec,El,Ej,phi_dc,omega_01,alpha, base_ex, base_size)



    #sanity
    #Ht  =get_Hamiltonian(Ec,El,Ej,phi_dc,omega_01,alpha, base_ex, base_size,expansion_order,only="ElEj")
    #print(np.sum(np.dot(np.dot(eigvecs[:,0].T.conj(),Ht),eigvecs[:,0])))

    
    H0, eigvals, eigvecs, basisTransform = sol(Ec,El,Ej,phi_dc,omega_01,alpha, base_ex, base_size,expansion_order)

    #diagonalized_H = np.dot(np.dot(basisTransform.T, H0), basisTransform)
    diagonalized_H = np.diag(eigvals)
    #print(diagonalized_H-diagonalized_H.T.conj())
    
    #print(diagonalized_H-diagonalized_H.T.conj())
    return diagonalized_H, eigvecs, basisTransform, eigvals, base_ex, base_size, Ec, El, Ej
    

from matplotlib import pyplot as plt
from matplotlib import colors

def init_qubit(Ec,El,Ej,phi_dc,omega_01,alpha,c_ops,Lambdas, base_ex=np.pi*4, base_size=1001,expansion_order=50, truncation=20,optimizebasis=True,Ec_IC=None, El_IC=None, Ej_IC=None):
    global H0
    global n_opp
    global phi_opp
    #global c_ops
    #check cops
    if El == 0: periodic = True
    else: periodic = False
    for i in range(len(c_ops)):
        if hasattr(c_ops[i],"full"):#qobj
            c_ops[i] = c_ops[i].full()[:2,:2]
        c_ops_tmp = np.array(c_ops[i])
        c_ops[i] = np.zeros((truncation,truncation),dtype=np.complex128)
        c_ops[i][:len(c_ops_tmp),:len(c_ops_tmp)] = c_ops_tmp

    H0_diag, eigenvecs, basisTransform, eigenvals, new_ex, new_size, Ec, El, Ej = get_diag_Hamiltonian(Ec,El,Ej,phi_dc,omega_01,alpha, base_ex, base_size,expansion_order, truncation=truncation,periodic=periodic,optim=optimizebasis,Ec_IC=Ec_IC, El_IC=El_IC, Ej_IC=Ej_IC)
    t_g = Lambdas/(H0_diag[1,1]-H0_diag[0,0])*2*np.pi
    base_ex = new_ex
    base_size = new_size
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
    basisTransform = cupy.asarray(basisTransform)
    #n1 = np.diag(n)
    #n2 = np.diag(n,1)
    #n3 = np.diag(n,-1)
    #n = cupyx.scipy.sparse.diags([n1,n2,n3], [0,1,-1], format="csr")
    n = cupyx.scipy.sparse.csr_matrix(n)
    #phi1 = np.diag(phi)
    #phi2 = np.diag(phi,1)
    #phi3 = np.diag(phi,-1)
    #phi = cupyx.scipy.sparse.diags([phi1,phi2,phi3], [0,1,-1], format="csr")
    phi = cupyx.scipy.sparse.csr_matrix(phi)
    #n = cupy.dot(cupy.dot(basisTransform.T.conj(), n), basisTransform)
    n = basisTransform.T.conj() @ n @ basisTransform
    #print(n-n.T.conj())
    #phi = cupy.dot(cupy.dot(basisTransform.T.conj(), phi), basisTransform)
    phi = basisTransform.T.conj() @ phi @ basisTransform

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
    tmp = n_opp[:truncation,:truncation].get()
    del n_opp
    n_opp = tmp
    tmp = phi_opp[:truncation,:truncation].get()
    del phi_opp
    phi_opp = tmp
    eigenvecs = eigenvecs[:,:truncation]
    tmp = basisTransform[:,:truncation].get()
    del basisTransform
    basisTransform = tmp
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

    return H0, n_opp, phi_opp, c_ops, t_g, base_ex, base_size, Ec, El, Ej



#init_qubit(1.3,0.59,5.71,np.pi)