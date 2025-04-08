import qutip as qt
import numpy as np
import sympy as sp
from matplotlib import pyplot as plt

config = {}
with open("config.txt", "r") as f:
    for line in f.readlines():
        line = line.strip()
        config[line.split(":")[0]] = eval(line.split(":")[1])

def factor_matrix_function(matrix):
    variable = sp.symbols('t')
    function = sp.Function('f')(variable)
    M = sp.Matrix(matrix)
    # Flatten the matrix into a list for GCD calculation
    elements = M.tolist()
    flat_elements = [item for sublist in elements for item in sublist]

    # Compute the greatest common divisor (GCD)
    common_factor = sp.gcd(flat_elements)

    # Divide each element by the common factor to get the scaling matrix
    scale_matrix = M.applyfunc(lambda expr: expr / common_factor)

    return scale_matrix, common_factor


class GateParams:
    def __init__(self, param_dict):
        self.__dict__.update(param_dict)

        #check kwargs wether they are of (lambda) function type
        """for key, value in self.__dict__.items():
            if not callable(value):
                print(f"Warning: {key} is not a function")"""
        
        #check if there is anything to calibrate:
        self.is_calibrated = {}
        keys = list(self.__dict__.keys())
        for key in keys:
            if key.startswith("calibrate_"):
                if "_Z" in key:
                    if self.__dict__[key] == True:
                        self.Z_amp = 0
                        self.is_calibrated["Z"] = False
                elif "_Omega" in key:
                    if self.__dict__[key] == True:
                        self.Omega_amp = 1
                        self.is_calibrated["Omega"] = False
    """    def assert_calibration(self, key, value):
        if key == "Z":
            self.Z_amp = value
            self.is_calibrated["Z"] = True
        elif key == "Omega":
            self.Omega_amp = value
            self.is_calibrated["Omega"] = True
        else:
            raise ValueError("Key not found")
        return self"""
    def assert_H0(self, H0):
        self.H0 = H0
        return self
    
    def assert_opps(self, n_opp, phi_opp):
        self.n_opp = n_opp
        self.phi_opp = phi_opp
        return self
    
    def compile_as_Qobj(self):
        #takes the H0, opps and parameters and makes actual function/objects out of them
        #must be divied up into a matrix of the static part, and function for the time-dependent part
        self.H0_bare = np.array(self.H0).astype(np.complex128)
        self.H0 = qt.Qobj(self.H0)

        self.polarization = np.array(self.polarization).astype(np.complex128)
        #carrier = lambda t: np.exp(-1j*self.omega_d*t)*self.polarization
        #self.function = lambda t: np.sum([(self.envelope(t)*carrier(t))[i]*np.array([self.n_opp, self.phi_opp])[i] for i in range(2)], axis=0)

        carrier = lambda t: np.exp(-1j*self.omega_d*t)
        scaling = 1
        if hasattr(self, 'Omega_amp'):
            scaling = float(self.Omega_amp)
        if isinstance(self.envelope, sp.Piecewise):
            ts = sp.Symbol('t')
            carrier = sp.exp(-1j*self.omega_d*ts)
            self.function = self.envelope*carrier*scaling
        else:
            self.function = lambda t: self.envelope(t)*carrier(t)*scaling
        #print(self.n_opp-self.n_opp.T.conj())
        #print(self.phi_opp-self.phi_opp.T.conj())
        self.matrixelem_n = qt.Qobj(self.n_opp*self.polarization[0])
        self.matrixelem_phi = qt.Qobj(self.phi_opp*self.polarization[1])
        
        return self
    

    def transform_2_rotating_frame(self,t0,omega_01):#!doublecheck this one (does give non-unitary matrix)
        global config
        #self.H0 = 0.5*(self.H0 + self.H0.dag())#!temp
        #self.matrixelem = 0.5*(self.matrixelem + self.matrixelem.dag())#!temp
        #print(self.H0.full()-self.H0.full().T.conj())
        #print(self.H0.full()-self.H0.full().T.conj())
        #print(self.matrixelem.full()-self.matrixelem.full().T.conj())
        #print(self.matrixelem.full()-self.matrixelem.full().T.conj())

        #def constituents
        me_11 = np.zeros((len(self.H0.full()),len(self.H0.full())),dtype=np.complex128)
        me_00 = np.zeros((len(self.H0.full()),len(self.H0.full())),dtype=np.complex128)
        me_I = np.eye(len(self.H0.full()))
        me_00[0,0], me_11[1,1] = 1, 1
        me_00, me_11, me_I = qt.Qobj(me_00), qt.Qobj(me_11), qt.Qobj(me_I)
        #omega_01 = -omega_01
        #the transform matrix
        unitary = lambda t: (np.exp(1j*omega_01*t)-1)*me_11 + me_I
        i_unitary = lambda t: (np.exp(-1j*omega_01*t)-1)*me_11 + me_I

        #time-part of transform
        time_part = -omega_01*me_11

        #H0 part
        #print(self.H0.full()-self.H0.full().T.conj())
        self.H0_evol = lambda t: unitary(t)*self.H0*i_unitary(t) + time_part
        #print(self.H0_evol(0).full())
        #print(self.H0_evol(5).full())
        #print(self.H0_evol(0).full()-self.H0_evol(0).full().T.conj())
        #print(self.H0_evol(5).full()-self.H0_evol(5).full().T.conj())
        
        #H_int part
        #print(self.matrixelem.full()-self.matrixelem.full().T.conj())
        #if self.funcbuffer not exists
        if not hasattr(self, 'funcbuffer'):
            self.funcbuffer = []
        if not self.known_t0:
            ts = sp.symbols('t')
            t0s = sp.Symbol('t_0')
            func = self.function
            func = func.subs(t0s, t0)
            #func = self.function.subs({self.t_0: t0})
            #convert to numpy piecewise
            func = sp.lambdify(ts, func, modules=["numpy"])
        else:
            func = self.function
        #print(func(0))
        #print(func(5))
        self.funcbuffer.append(func)

        func1 = lambda t: (unitary(t)*self.matrixelem_n*i_unitary(t)*self.funcbuffer[-1](t))
        #plt.plot(np.linspace(0, 5, 100), [func1(t).full()[0][1] for t in np.linspace(0, 5, 100)])
        #plt.savefig("temp/envelope.png")
        #plt.clf()
        func2 = lambda t: (unitary(t)*self.matrixelem_phi*i_unitary(t)*self.funcbuffer[-1](t))
        #plt.plot(np.linspace(0, 5, 100), [func2(t).full()[0][1] for t in np.linspace(0, 5, 100)])
        #plt.savefig("temp/envelope.png")
        #plt.clf()
        self.function_HI = lambda t: 0.5*(func1(t)+func1(t).dag()) + 0.5*(func2(t)+func2(t).dag())# + time_part
        #plt.plot(np.linspace(0, 5, 100), [self.function_HI(t).full()[0][1] for t in np.linspace(0, 5, 100)])
        #plt.savefig("temp/envelope.png")
        #plt.clf()
        #print(self.function_HI_t(0).full()-self.function_HI_t(0).full().T.conj())
        #print(self.function_HI_t(5).full()-self.function_HI_t(5).full().T.conj())
        
        #self.function_HI = lambda t: 0.5*(self.function_HI_t(t)+self.function_HI_t(t).dag())#symmetrize#!temp
        #print(self.function_HI(0).full()-self.function_HI(0).full().T.conj())
        #print(self.function_HI(5).full()-self.function_HI(5).full().T.conj())
        #print(self.function(5)*np.conjugate(self.function(5)))
        
        #H = H0 + H_int
        def H(t):
            mat = self.H0_evol(t) + self.function_HI(t)
            if config["H_log"]:
                with open("H_log.txt", "a") as f:
                    f.write(f"{t}; {mat.full()[0][1]}\n")
            return mat
        self.function_H = lambda t: H(t)
        #self.function_H = lambda t: self.H0_evol(t)+self.function_HI(t)#one could argue that the code would be optimizable by seperating function and matrix, but I would argue (or prove): this is not possible for general hammiltonians
        
    
        #print(self.function_H(0).full()-self.function_H(0).full().T.conj())
        #print(self.function_H(5).full()-self.function_H(5).full().T.conj())
        
        return self.H0_bare, qt.QobjEvo(self.function_H)

    
    
    def get_QuTiP_compile(self):
        #done already
        #return qt.QobjEvo(self.function_H)
        #first H0
        H0 = qt.Qobj(self.H0)
        #use sympy to factor out the time-dependent part
        function_HI = lambda t: qt.Qobj(self.function_HI(t))
        H1 = qt.QobjEvo(function_HI)
        #H,func = factor_matrix_function(self.function_HI)
        #H1 = qt.QobjEvo([H, func])
        return [H0, H1]