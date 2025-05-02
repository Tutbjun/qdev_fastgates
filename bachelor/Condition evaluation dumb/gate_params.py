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
    elif gate == "corotating_xy":
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
            "calibrate_Z": False,
            "name": "corotating_xy",
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
    elif gate == "commensurate_x_virt_z":
        theta_a = 0
        theta_b = 0#from resonant pulse condition
        dt_0 = -0.5*sp.Symbol('t_g')
        t_g = sp.Symbol('t_g')
        #varphi = 0
        omega_d = sp.Symbol('omega_{01}')
        nt_0 = sp.Symbol('n')*np.pi/omega_d
        t_0 = nt_0+dt_0
        #tm = sp.Symbol('t_m')
        #t = tm+t_0+0.5*t_g
        t = sp.Symbol('t')
        tm = t-t_0-0.5*t_g
        
        envelope = sp.Piecewise((0.5*(sp.cos(sp.pi*tm/t_g)**2), (tm>=-0.5*t_g) & (tm<=0.5*t_g)), (0,True))

        return po.GateParams({
            "omega_d": omega_d,
            "known_t0": True,
            "calibrate_Omega": True,
            "calibrate_VZ": True,
            "name": "commensurate_x_virt_z",
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
    elif gate == "magnus1_x_virt_z":
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
        lambda_initial = (2*omega_d)**-1
        envelope_inphase = sp.Piecewise((Omega*(1-sp.cos(2*sp.pi*tm/t_g)), (tm>=0*t_g) & (tm<=t_g)), (0,True))
        envelope_quad = sp.Piecewise((Omega*lambda_*2*sp.pi/t_g*sp.sin(2*sp.pi*tm/t_g), (tm>=0*t_g) & (tm<=t_g)), (0,True))
        return po.GateParams({
            "omega_d": omega_d,
            "known_t0": False,
            "calibrate_Omega": True,
            "initial_Omega": Omega_initial,
            "calibrate_VZ": True,
            "calibrate_lambda": True,
            "initial_lambda": lambda_initial,
            "name": "magnus1_x_virt_z",
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