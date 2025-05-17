import qutip as qt
import qutip_cupy
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
        self.VZ_amp = 0
        self.is_calibrated = {}
        keys = list(self.__dict__.keys())
        for key in keys:
            if key.startswith("calibrate_"):
                if "_VZ" in key:
                    if self.__dict__[key] == True:
                        self.VZ_amp = 0
                        self.VZ_amp_buffer = []
                        self.VZ = [[1,0],[0,1]]
                        self.is_calibrated["VZ"] = False
                elif "_Omega" in key:
                    if self.__dict__[key] == True:
                        self.Omega_amp = 1
                        if "initial_Omega" in param_dict:
                            self.Omega_amp = param_dict["initial_Omega"]
                        self.is_calibrated["Omega"] = False
                    else:
                        self.Omega_amp = param_dict["initial_Omega"]
                elif "_lambda" in key:
                    if self.__dict__[key] == True:
                        self.lambda_amp = 1
                        if "initial_lambda" in param_dict:
                            self.lambda_amp = param_dict["initial_lambda"]
                        self.is_calibrated["lambda"] = False
                elif "_dt0" in key:
                    if self.__dict__[key] == True:
                        self.dt0 = 0
                        if "initial_dt0" in param_dict:
                            self.dt0_amp = param_dict["initial_dt0"]
                        self.is_calibrated["dt0"] = False


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
        print(self.polarization)
        #carrier = lambda t: np.exp(-1j*self.omega_d*t)*self.polarization
        #self.function = lambda t: np.sum([(self.envelope(t)*carrier(t))[i]*np.array([self.n_opp, self.phi_opp])[i] for i in range(2)], axis=0)

        carrier = lambda t: np.exp(-1j*self.omega_d*t)
        carrier_inphase = lambda t: carrier(t)
        carrier_quad = lambda t: carrier(t)*np.exp(-1j*sp.pi/2)
        scaling = 1
        lambda_ = 1
        if hasattr(self, 'Omega_amp'):
            if isinstance(self.Omega_amp, sp.Basic):
                scaling = self.Omega_amp
            else:
                scaling = float(self.Omega_amp)
        if hasattr(self, 'lambda_amp'):
            lambda_ = float(self.lambda_amp)
        if hasattr(self, 'envelope'):
            if isinstance(self.envelope, sp.Piecewise):
                ts = sp.Symbol('t')
                carrier = sp.exp(-1j*self.omega_d*ts)
                self.function = self.envelope*carrier*scaling
                #self.function = self.function.subs(sp.Symbol('omega_{01}'), self.H0_bare[1,1]-self.H0_bare[0,0])
            else:
                self.function = lambda t: self.envelope(t)*carrier(t)*scaling
        elif hasattr(self, 'envelope_inphase') and hasattr(self, 'envelope_quad'):
            if isinstance(self.envelope_inphase, sp.Piecewise):
                ts = sp.Symbol('t')
                carrier_inphase = sp.exp(-1j*self.omega_d*ts)
                #plot carrier_inphase
                """cip_dummy = sp.lambdify(sp.Symbol("t"),carrier_inphase.subs(sp.Symbol("omega_{01}"), 1), modules=["numpy"])
                plt.plot(np.linspace(0, 10, 100), [cip_dummy(t) for t in np.linspace(0, 10, 100)])
                plt.savefig("temp/1.png")
                plt.clf()"""
                carrier_quad = sp.exp(-1j*self.omega_d*ts)*sp.exp(-1j*sp.pi/2)
                """cq_dummy = sp.lambdify(sp.Symbol("t"),carrier_quad.subs(sp.Symbol("omega_{01}"), 1), modules=["numpy"])
                plt.plot(np.linspace(0, 10, 100), [cq_dummy(t) for t in np.linspace(0, 10, 100)])
                plt.savefig("temp/2.png")
                plt.clf()"""
                env_inphase = self.envelope_inphase.subs(sp.Symbol("Omega"), self.Omega_amp).subs(sp.Symbol("lambda"), self.lambda_amp)
                """envip_dummy = sp.lambdify(sp.Symbol("t"),env_inphase.subs(sp.Symbol("t_0"), 0).subs(sp.Symbol("t_g"), 5), modules=["numpy"])
                plt.plot(np.linspace(0, 10, 100), [envip_dummy(t) for t in np.linspace(0, 10, 100)])
                plt.savefig("temp/3.png")
                plt.clf()"""
                env_quad = self.envelope_quad.subs(sp.Symbol("Omega"), self.Omega_amp).subs(sp.Symbol("lambda"), self.lambda_amp)
                """envq_dummy = sp.lambdify(sp.Symbol("t"),env_quad.subs(sp.Symbol("t_0"), 0).subs(sp.Symbol("t_g"), 5), modules=["numpy"])
                plt.plot(np.linspace(0, 10, 100), [envq_dummy(t) for t in np.linspace(0, 10, 100)])
                plt.savefig("temp/4.png")
                plt.clf()"""
                self.function = env_inphase*carrier_inphase + env_quad*carrier_quad
                """func_dummy = sp.lambdify(sp.Symbol("t"),self.function.subs(sp.Symbol("t_0"), 0).subs(sp.Symbol("t_g"), 20).subs(sp.Symbol("omega_{01}"), 1), modules=["numpy"])
                plt.plot(np.linspace(0, 30, 100), [func_dummy(t) for t in np.linspace(0, 30, 100)])
                plt.savefig("temp/5.png")
                plt.clf()"""
                #self.function = self.function.subs(sp.Symbol('omega_{01}'), self.H0_bare[1,1]-self.H0_bare[0,0])
            else:
                raise NotImplementedError
        #print(self.n_opp-self.n_opp.T.conj())
        #print(self.phi_opp-self.phi_opp.T.conj())
        #self.matrixelem_n = qt.Qobj(self.n_opp)#*self.polarization[1])
        #matrixelem_phi = qt.Qobj(self.phi_opp)#*self.polarization[0])
        #print(self.matrixelem_n.full())
        #print(matrixelem_phi.full())
        
        return self

    def transform_2_rotating_frame(self,t_g,omega_01,t0=None,n=None,dt0=None):
        #attempt 2, this time simpler
        me_11 = np.zeros((len(self.H0.full()),len(self.H0.full())),dtype=np.complex128)
        me_00 = np.zeros((len(self.H0.full()),len(self.H0.full())),dtype=np.complex128)
        me_I = np.eye(len(self.H0.full()))
        me_00[0,0], me_11[1,1] = 1, 1
        #me_00, me_11, me_I = qt.Qobj(me_00), qt.Qobj(me_11), qt.Qobj(me_I)
        unitary = lambda t: (np.exp(1j*omega_01*t))*me_11 + me_00
        i_unitary = lambda t: (np.exp(-1j*omega_01*t))*me_11 + me_00

        #then construct the H0 and HI
        if not hasattr(self, 'funcbuffer'):
            self.funcbuffer = []
        #if not self.known_t0:
        if self.function == 0:
            func = lambda t: 0
        else:
            ts = sp.symbols('t')
            t0s = sp.Symbol('t_0')
            t_gs = sp.Symbol('t_g')
            omega_01s = sp.Symbol('omega_{01}')
            func = self.function
            if t0 != None: func = func.subs(t0s, t0)
            if n != None: func = func.subs(sp.Symbol('n'), n)
            func = func.subs(t_gs, t_g)
            func = func.subs(omega_01s, omega_01)
            if dt0 != None:
                func = func.subs(sp.Symbol('dt_0'), dt0)
            #func = self.function.subs({self.t_0: t0})
            #convert to numpy piecewise
            func = sp.lambdify(ts, func, modules=["numpy"])
        """else:
            func = self.function
            if func == 0:
                func = lambda t: 0"""
        matrixelem_n = qt.Qobj(self.n_opp)
        matrixelem_phi = qt.Qobj(self.phi_opp)
        #print(matrixelem_n.full())
        #print(matrixelem_phi.full())
        #matrixelem_n = 0.5*(matrixelem_n + matrixelem_n.dag())#!temp
        #matrixelem_phi = 0.5*(matrixelem_phi + matrixelem_phi.dag())#!temp
        #normalize
        #matrixelem_n = matrixelem_n/np.abs(matrixelem_n[0][1])
        #matrixelem_phi = matrixelem_phi/np.abs(matrixelem_phi[0][1])
        func1_t = lambda t: np.real(func(t)*self.polarization[1])
        func2_t = lambda t: np.real(func(t)*self.polarization[0])
        """plt.plot(np.linspace(0, 30, 100), [func1_t(t) for t in np.linspace(0, 30, 100)])
        plt.plot(np.linspace(0, 30, 100), [func2_t(t) for t in np.linspace(0, 30, 100)])
        plt.savefig("temp/envelope.png")
        plt.clf()
        plt.close('all')"""
        #func1 = lambda t: matrixelem_n*func1_t(t)
        #func2 = lambda t: matrixelem_phi*func2_t(t)
        func1 = lambda t: self.n_opp*func1_t(t)
        func2 = lambda t: self.phi_opp*func2_t(t)
        #print(matrixelem_n.full())#! not unitary
        #print(matrixelem_phi.full())
        #print(self.H0.full())
        H0 = self.H0.full()
        H_evol = lambda t: H0 + func1(t) + func2(t)
        #plot this
        #T = np.linspace(0, 20, 100)
        #r = [H_evol(t).full()[0][1].real for t in T]
        #i = [H_evol(t).full()[0][1].imag for t in T]
        #r2 = [H_evol(t).full()[1][0].real for t in T]
        #i2 = [H_evol(t).full()[1][0].imag for t in T]
        """plt.plot(T, r)
        plt.plot(T, i)
        plt.plot(T, r2)
        plt.plot(T, i2)
        plt.savefig("temp/envelope.png")
        plt.clf()
        plt.close('all')"""


        #then, finally, do the transformation
        time_part = -omega_01*me_11
        #H_transformed = lambda t: unitary(t)*H_evol(t)*i_unitary(t) + time_part
        def H_transformed(t):
            v = unitary(t)@H_evol(t)@i_unitary(t)
            mat = v + time_part
            mat[1][1] += np.random.uniform(-1,1)*1e-9#this is the extend to wich we trust it
            r = np.random.uniform(-1,1)*1e-9
            t = np.random.uniform(0,2*np.pi)
            mat[0][1] += r*np.exp(1j*t)
            mat[1][0] += r*np.exp(-1j*t)
            if config["H_log"]:
                #matf = mat.full()
                with open("temp/H_log.txt", "a") as f:
                    f.write(f"{t}; {mat[0][1]}\n")
                with open("temp/H_log11.txt", "a") as f:
                    f.write(f"{t}; {mat[1][1]-mat[0][0]}\n")
            return qt.Qobj(mat)
        #T = np.linspace(0, 20, 100)
        #r = [H_transformed(t).full()[0][1].real for t in T]
        #i = [H_transformed(t).full()[0][1].imag for t in T]
        #r2 = [H_transformed(t).full()[1][0].real for t in T]
        #i2 = [H_transformed(t).full()[1][0].imag for t in T]
        """plt.plot(T, r)
        plt.plot(T, i)
        plt.plot(T, r2)
        plt.plot(T, i2)
        plt.savefig("temp/envelope.png")
        plt.clf()
        plt.close('all')"""

        return self.H0_bare, qt.QobjEvo(H_transformed)
    
    
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