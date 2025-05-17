from utils import param_object as po
import numpy as np
import sympy as sp


def get_gate_params(gate="demo"):
    if gate == "demo_x":

        #omega_d = 0.254
        #envelope = cos of amplitude 0.5
        #polarization = 0
        #t_g = 1
        theta_a = np.pi/4
        theta_b = np.pi*0
        t_g = 1
        #omega_d = 0.51740288829156995#0.254
        #omega_d = 0.04952
        omega_d = 0.22253833
        t_0 = sp.Symbol('t_0')
        t = sp.Symbol('t')
        t_g = sp.Symbol('t_g')
        known_t0 = False
        #envelope = lambda t: np.piecewise(t,[t<t_0, (t>=t_0)*(t<=t_g+t_0), t>t_g+t_0], [0,0.5*(1-np.sin(np.pi*(t-t_0)/t_g)**2),0]) #!should this be t0 dependant?
        envelope = sp.Piecewise((0.5*(sp.sin(sp.pi*(t-t_0)/t_g)**2), (t>=t_0) & (t<=t_g+t_0)), (0,True))
        #    [0,,0], [t<t_0, (t>=t_0)*(t<=t_g+t_0), t>t_g+t_0])
        #!should this be t0 dependant?
        #polarization = [1,0]
        return po.GateParams({
            "omega_d": omega_d,
            "envelope": envelope,
            "known_t0": known_t0,
            "calibrate_Omega": True,
            "calibrate_Z": False,
            "name": "demo_x",
            "t_0": t_0,
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(-1j*theta_b)),
            #"t_g": t_g,
            "ideal": [[0,1j],[1j,0]],
            "c_ops": [
                np.array([[0,158],[0,0]]),
                np.array([[0,1],[0,0]])
            ]
        })
    elif gate == "RWA_x_nooptim":
        #omega_d = 0.254
        #envelope = cos of amplitude 0.5
        #polarization = 0
        #t_g = 1
        theta_a = 0
        theta_b = 0

        t_g = 1
        #omega_d = 0.51740288829156995#0.254
        #omega_d = 0.04952
        omega_d = sp.Symbol('omega_{01}')
        t_0 = sp.Symbol('t_0')
        t = sp.Symbol('t')
        t_g = sp.Symbol('t_g')
        envelope = sp.Piecewise(((sp.sin(sp.pi*(t-t_0)/t_g)**2), (t>=t_0) & (t<=t_g+t_0)), (0,True))
        return po.GateParams({
            "omega_d": omega_d,
            "envelope": envelope,
            "Omega_eq": np.pi/(t_g),
            "known_t0": False,
            "calibrate_Omega": False,
            "initial_Omega": np.pi/(t_g),
            "calibrate_Z": False,
            "name": "RWA_x",
            "t_0": t_0,
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(-1j*theta_b)),
            #"t_g": t_g,
            "ideal": [[1/np.sqrt(2),-1j/np.sqrt(2)],[-1j/np.sqrt(2),1/np.sqrt(2)]],#x gate
        })
    elif "corotating_xy" in gate:
        theta_a = np.pi/4
        theta_b = np.pi/2
        omega_d = sp.Symbol('omega_{01}')#resonant
        t_0 = sp.Symbol('t_0')
        t_g = sp.Symbol('t_g')
        t = sp.Symbol('t')
        known_t0 = False
        calibrate_Omega = True
        Omega_eq = None
        envelope = sp.Piecewise((0.5*(sp.sin(sp.pi*(t-t_0)/t_g)**2), (t>=t_0) & (t<=t_g+t_0)), (0,True))
        name = "corotating_xy"
        if "nooptim" in gate:
            calibrate_Omega = False
            Omega_eq= np.pi/t_g*np.sqrt(2)
            name = "corotating_xy_nooptim"
        
        return po.GateParams({
            "omega_d": omega_d,
            "envelope": envelope,
            "known_t0": known_t0,
            "calibrate_Omega": calibrate_Omega,
            "Omega_eq": Omega_eq,
            "initial_Omega": Omega_eq,
            "calibrate_Z": False,
            "name": name,
            "t_0": t_0,
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(-1j*theta_b)),
            "ideal": [[1/np.sqrt(2),-1j/np.sqrt(2)],[-1j/np.sqrt(2),1/np.sqrt(2)]],#x gate
        })
    elif gate == "corotating_xy_virt_z":
        theta_a = np.pi/4
        theta_b = np.pi/2
        omega_d = sp.Symbol('omega_{01}')#resonant
        t_0 = sp.Symbol('t_0')
        t_g = sp.Symbol('t_g')
        t = sp.Symbol('t')
        known_t0 = False
        envelope = sp.Piecewise((0.5*(sp.sin(sp.pi*(t-t_0)/t_g)**2), (t>=t_0) & (t<=t_g+t_0)), (0,True))
        return po.GateParams({
            "omega_d": omega_d,
            "envelope": envelope,
            "known_t0": known_t0,
            "calibrate_Omega": True,
            "initial_Omega": np.pi/(t_g),
            "calibrate_VZ": True,
            "name": "corotating_xy",
            "t_0": t_0,
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(-1j*theta_b)),
            "ideal": [[1/np.sqrt(2),-1j/np.sqrt(2)],[-1j/np.sqrt(2),1/np.sqrt(2)]],#x gate
        })
    elif gate == "simple_x":
         #omega_d = 0.254
        #envelope = cos of amplitude 0.5
        #polarization = 0
        #t_g = 1
        theta_a = np.pi*0
        theta_b = np.pi*0
        #t_g = 1
        #omega_d = 0.51740288829156995#0.254
        #omega_d = 0.04952
        omega_d = sp.Symbol('omega_{01}')#resonant
        t_0 = sp.Symbol('t_0')
        t_g = sp.Symbol('t_g')
        t = sp.Symbol('t')
        known_t0 = False
        #envelope = lambda t: np.piecewise(t,[t<t_0, (t>=t_0)*(t<=t_g+t_0), t>t_g+t_0], [0,0.5*(1-np.sin(np.pi*(t-t_0)/t_g)**2),0]) #!should this be t0 dependant?
        envelope = sp.Piecewise((0.5*(sp.sin(sp.pi*(t-t_0)/t_g)**2), (t>=t_0) & (t<=t_g+t_0)), (0,True))
        #    [0,,0], [t<t_0, (t>=t_0)*(t<=t_g+t_0), t>t_g+t_0])
        #!should this be t0 dependant?
        #polarization = [1,0]
        return po.GateParams({
            "omega_d": omega_d,
            "envelope": envelope,
            "known_t0": known_t0,
            "calibrate_Omega": True,
            "calibrate_Z": False,
            "initial_Omega": np.pi/(t_g),
            "name": "corotating_xy",
            "t_0": t_0,
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(-1j*theta_b)),
            #"t_g": t_g,
            #"ideal": [[1/np.sqrt(2),-1/np.sqrt(2)],[1/np.sqrt(2),1/np.sqrt(2)]],#y gate
            "ideal": [
                [1/np.sqrt(2),-1j/np.sqrt(2)],
                [-1j/np.sqrt(2),1/np.sqrt(2)]
            ]
            #"ideal": [
            #    [1/np.sqrt(2),-0.5+0.5j],
            #    [0.5+0.5j,1/np.sqrt(2)]
            #]
        })
    elif "commensurate_x_virt_z" in gate:
        #Pi*(-a^2*omega^2 + Pi^2)*omega/(-2*a^3*omega^3 + 2*Pi^2*a*omega + 2*Pi^2*sin(omega*a))
        theta_a = 0
        theta_b = 0#from resonant pulse condition
        dt_0_init = -0.5*sp.Symbol('t_g')
        dt_0 = sp.Symbol('dt_0')
        t_g = sp.Symbol('t_g')
        #varphi = 0
        omega_d = sp.Symbol('omega_{01}')
        nt_0 = sp.Symbol('n')*np.pi/omega_d
        t_0 = nt_0+dt_0
        #tm = sp.Symbol('t_m')
        #t = tm+t_0+0.5*t_g
        t = sp.Symbol('t')
        tm = t-t_0-0.5*t_g
        calibrate_Omega = True
        calibrate_dt0 = True
        #Omega_eq= np.pi*(-t_g**2*omega_d**2 + np.pi**2)*omega_d/(-2*t_g**3*omega_d**3 + 2*np.pi**2*t_g*omega_d + 2*np.pi**2*sp.sin(omega_d*t_g))*4#maple formula
        Omega_eq = 3.25*(-4*np.pi**2*t_g**2*omega_d**3 + np.pi**4*omega_d)/((8*1j*omega_d**2*t_g**2*np.pi - 2*1j*np.pi**3 + 4*np.pi**2*t_g*omega_d)*sp.exp(2*1j*omega_d*(dt_0 + t_g)) + (-8*1j*omega_d**2*t_g**2*np.pi + 2*1j*np.pi**3 + 4*np.pi**2*t_g*omega_d)*sp.exp(2*1j*omega_d*dt_0) + 4*omega_d*t_g*(np.pi + 2)*(-2*t_g*omega_d + np.pi)*(2*t_g*omega_d + np.pi))#maple formula
        dt0_eq = -0.5*t_g
        dt0_eq = None
        name = "commensurate_x_virt_z"
        if "nooptim" in gate:
            calibrate_Omega = False
            calibrate_dt0 = True
            name += "_nooptim"

        envelope = sp.Piecewise((2*(sp.cos(sp.pi*tm/t_g)**2), (tm>=-0.5*t_g) & (tm<=0.5*t_g)), (0,True))
        """from copy import copy
        import matplotlib.pyplot as plt
        env_cop = copy(envelope)
        env_cop = env_cop.subs(t_g, 1).subs(sp.Symbol('n'), 0)
        X = np.linspace(-0.5, 0.5, 100)
        Y = np.array([env_cop.subs(t, x) for x in X])
        plt.plot(X, Y)
        plt.title("Envelope")
        plt.xlabel("t")
        plt.ylabel("envelope")
        plt.grid()
        plt.savefig("envelope_tmp.png")
        plt.show()"""
        return po.GateParams({
            "omega_d": omega_d,
            "known_t0": True,
            "calibrate_Omega": calibrate_Omega,
            "calibrate_dt0": calibrate_dt0,
            "initial_dt0": dt_0_init,
            "initial_Omega": Omega_eq,
            "calibrate_VZ": True,
            "name": name,
            "Omega_eq": Omega_eq,
            "dt0_eq": dt0_eq,
            "t_0": t_0,
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(-1j*theta_b)),
            "envelope": envelope,
            #"carrier": sp.cos(omega_d*t),
            "ideal": [[1/np.sqrt(2),-1j/np.sqrt(2)],[-1j/np.sqrt(2),1/np.sqrt(2)]],#x gate
            "t_g": t_g,
            "c_ops": [
                np.array([[0,158],[0,0]]),
                np.array([[0,1],[0,0]])
            ]
        })
    elif "FAST-MAGNUS" in gate:
        c_arr = [sp.Symbol(f'c_{n}') for n in range(1, 5)]
        envelope = sp.Piecewise(
            (
                sum(

                )
                (t>=t_0) & (t<=t_g+t_0))
            , (0,True))
    elif "magnus1_x_virt_z" in gate:
        phi = 0
        omega_d = sp.Symbol('omega_{01}')
        theta_a = 0
        theta_b = 0
        Omega = sp.Symbol('Omega')
        t_g = sp.Symbol('t_g')
        t_0 = sp.Symbol('t_0')
        t = sp.Symbol('t')
        tm = t-t_0
        Omega_initial = np.pi/t_g/2
        lambda_ = sp.Symbol('lambda')
        lambda_initial = -0.5*(2*omega_d)**-1
        envelope_inphase = sp.Piecewise((Omega*(1-sp.cos(2*sp.pi*tm/t_g)), (tm>=0*t_g) & (tm<=t_g)), (0,True))
        envelope_quad = sp.Piecewise((Omega*lambda_*2*sp.pi/t_g*sp.sin(2*sp.pi*tm/t_g), (tm>=0*t_g) & (tm<=t_g)), (0,True))
        name = "magnus1_x_virt_z"
        calibrate_Omega = True
        calibrate_lambda = True
        Omega_eq = None
        if "nooptim" in gate:
            calibrate_Omega = False
            calibrate_lambda = False
            Omega_eq= np.pi/t_g/2
            name += "_nooptim"
        return po.GateParams({
            "omega_d": omega_d,
            "known_t0": False,
            "calibrate_Omega": calibrate_Omega,
            "initial_Omega": Omega_initial,
            "Omega_eq": Omega_eq,
            "calibrate_VZ": True,
            "calibrate_lambda": calibrate_lambda,
            "initial_lambda": lambda_initial,
            "lambda_eq": lambda_initial,
            "name": name,
            "t_0": t_0,
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(-1j*theta_b)),
            "envelope_inphase": envelope_inphase,
            "envelope_quad": envelope_quad,
            #"carrier": sp.cos(omega_d*t),
            "ideal": [[1/np.sqrt(2),-1j/np.sqrt(2)],[-1j/np.sqrt(2),1/np.sqrt(2)]],#x gate
            "t_g": t_g,
            "c_ops": [
                np.array([[0,158],[0,0]]),
                np.array([[0,1],[0,0]])
            ]
        })
    
    else:
        raise ValueError("Gate not found")