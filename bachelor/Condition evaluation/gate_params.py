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
            "polarization": (np.cos(theta_a), np.sin(theta_a)*np.exp(1j*theta_b)),
            "t_g": t_g,
            "ideal": [[0,1j],[1j,0]],
            "c_ops": [
                np.array([[0,158],[0,0]]),
                np.array([[0,1],[0,0]])
            ]
        })
    else:
        raise ValueError("Gate not found")