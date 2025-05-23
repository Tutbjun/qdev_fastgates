import adaptive.learner.learner2D
from utils import gate_evaluator as ge
from utils import gate_simulator as gs
from utils import param_object as po
import os
from matplotlib import pyplot as plt
import numpy as np
import pickle
from pathlib import Path
import time
from utils import qubit_init as qbi
import sympy as sp
from copy import deepcopy as copy
import adaptive
config = {}
with open("config.txt", "r") as f:
    for line in f.readlines():
        line = line.strip()
        config[line.split(":")[0]] = eval(line.split(":")[1])
class AdaptiveLearner:
    def conv_bounds(self, bounds, xs):
        #make a mask for the values that are non-fixed
        mask = np.array([b is not None for b in bounds])
        self.eval_mask = mask
        if len(xs) > 0:
            self.xs = xs[:, mask]
        self.bounds_actual = [b for i,b in enumerate(bounds) if mask[i]]
    def __init__(self, known_dps, bounds, framework="adaptive", truncation=2):
        #known_dps are expected to be a list of dictionaries
        ys = [dp["score"] for dp in known_dps]
        xs = [[dp["Ec"], dp["El"], dp["Ej"], dp['omega_01_target'], dp['alpha_target'], dp["phi_dc"],dp["Lambdas"],dp["L1"],dp["L2"]] for dp in known_dps]
        point_keys = ["Ec", "El", "Ej", "omega_01_target", "alpha_target", "phi_dc", "Lambdas", "L1", "L2"]
        self.ys = np.array(ys)
        self.xs = np.array(xs)
        self.point_keys = point_keys
        self.truncation = truncation

        #bounds are expected to be a list of tuples. If an ellement is of float type rather than tuple, it is accepted as a fixed value
        bnds = [b if isinstance(b, tuple) else None for b in bounds]
        if bnds[6] is not None:
            bnds[6] = (np.log10(bnds[6][0]), np.log10(bnds[6][1]))#t_g is expected to be in log scale
        fixed = [b if not isinstance(b, tuple) else None for b in bounds]
        if fixed[6] is not None:
            fixed[6] = (np.log10(fixed[6]), np.log10(fixed[6]))#t_g is expected to be in log scale
        self.bounds = bnds
        self.fixed = fixed
        self.drawn_points = 0

        #if any of the fixed are of array type, then multiple fixed points are expected, and the learner should have sub-learners for each fixed point
        dosublearners = False
        for i in range(len(fixed)):
            if isinstance(fixed[i], np.ndarray):
                dosublearners = True
                break
        self.sublearners = []
        if dosublearners:
            multi_fixed_indecies = []
            for i in range(len(fixed)):
                if isinstance(fixed[i], np.ndarray):
                    multi_fixed_indecies.append(i)
            fixed_point_axis = [fixed[i] for i in multi_fixed_indecies]
            grid_points = np.meshgrid(*fixed_point_axis)
            grid_points = np.array(grid_points).T.reshape(-1, len(multi_fixed_indecies))
            for i in range(len(grid_points)):
                #create a new learner for each fixed point
                new_bounds = copy(bounds)
                for j in range(len(multi_fixed_indecies)):
                    new_bounds[multi_fixed_indecies[j]] = grid_points[i][j]
                #create a new learner
                sublearner = AdaptiveLearner(known_dps, new_bounds, framework=framework)
                self.sublearners.append(sublearner)

        self.conv_bounds(bnds, self.xs)

        self.framework = framework
        if "adaptive" in framework:
            dummy_function = lambda x: None
            lossfunc = None
            if "area" in framework:
                lossfunc = adaptive.learner.learnerND.uniform_loss
            self.learner = adaptive.LearnerND(dummy_function, self.bounds_actual, loss_per_simplex=lossfunc)
            if len(self.xs) > 0:
                for x,y in zip(self.xs,self.ys):
                    self.learner.tell(x,y)
        elif framework == "random":
            import random
            self.learner = random.Random()
        elif framework == "bayesian":
            if len(self.xs[0]) >= 2:
                raise NotImplementedError("Not for multi-dim please")
            #self.xs = np.array([float(x) for x in self.xs])
            from skopt import Optimizer
            self.learner = Optimizer(self.bounds_actual, base_estimator="GP", acq_func="LCB", n_initial_points=10, random_state=None)
            if bnds[6] is not None:
                indx_t = [1 for i,b in enumerate(bnds[:6]) if b is not None]
                indx_t = int(np.array(indx_t).sum())
                self.xs.T[indx_t] = np.log10(self.xs.T[indx_t])
                #self.xs = np.log10(self.xs)
            if len(self.xs) > 0:
                print("Learning initial points")
                self.learner.tell(self.xs.tolist(), self.ys.tolist())
                """for x,y in zip(self.xs,self.ys):
                    x,y = [float(x)], [float(y)]
                    self.learner.tell(x,y)"""
        else:
            raise NotImplementedError(f"Framework {framework} is not implemented")
    def get_next_dp(self, N=1):
        if len(self.sublearners) > 0:
            points = []
            while len(points) < N:
                #get a point from  the sublearner with the least points
                min_points = np.inf
                min_index = -1
                for i in range(len(self.sublearners)):
                    if len(self.sublearners[i].xs) + self.sublearners[i].drawn_points < min_points:
                        min_points = len(self.sublearners[i].xs) + self.sublearners[i].drawn_points
                        min_index = i
                #get a point from the sublearner
                sublearner = self.sublearners[min_index]
                sub_point = sublearner.get_next_dp(1)
                points.append(sub_point[0])
            return points
        if "adaptive" in self.framework:
            points = self.learner.ask(N)
        elif self.framework == "bayesian":
            points = self.learner.ask(N)
        elif self.framework == "random":
            points = []
            for i in range(N):
                point = []
                k = -1
                for j in range(len(self.bounds)):
                    if self.bounds[j] is not None:
                        k += 1
                        cond = True
                        while cond:
                            pnt = self.learner.uniform(self.bounds[j][0], self.bounds[j][1])
                            cond = False
                            for i in range(self.xs.shape[0]):
                                if np.isclose(pnt, self.xs[i][k]):
                                    cond = True
                                    break
                            #cond = not cond
                        point.append(pnt)
                    else:
                        #point.append(self.fixed[j])
                        pass
                if len(point) == 1: point = (point[0],)
                #point = np.array(point)
                points.append(point)
            points = [points,np.full(N, np.inf)]
        else:
            raise NotImplementedError(f"Framework {self.framework} is not implemented")
        t_g_index = self.point_keys.index("Lambdas")
        for i in list(range(t_g_index)):
            if self.bounds[i] is None:#!new
                t_g_index -= 1
        if t_g_index > 0:
            points[0][t_g_index] = np.power(10, points[0][t_g_index])#convert t_g back to normal scale
        else:
            points= (np.power(10, points[0]),points[1])
        self.drawn_points += N
        #put fixed values in the points
        tpoints = []
        for point in points[0]:
            tpoint = []
            j = 0
            for i,f in enumerate(self.fixed):
                if f is not None:
                    tpoint.append(f)
                elif f is None and self.bounds[i] is None:
                    tpoint.append(None)
                else:
                    tpoint.append(point[j])
                    j += 1
            tpoints.append(tpoint)
        #convert these to list of dictionaries
        rpoints = []
        for point in tpoints:
            rpoint = {}
            for i, k in enumerate(self.point_keys):
                rpoint[k] = point[i]
            rpoints.append(rpoint)
        #covnert L1,L2 to decay opperators
        for i in range(len(rpoints)):
            rpoints[i] = self.convert_Ln_2_decay(rpoints[i])
        #name these qubits by their data
        #and check if this name has an index
        qubit_names = {}
        if os.path.exists("qubit_names.pickle"):
            with open("qubit_names.pickle", "rb") as f:#interprit as dict
                qubit_names = pickle.load(f)
        for i in range(len(rpoints)):
            name = f"qubit_{rpoints[i]['Ec']}_{rpoints[i]['El']}_{rpoints[i]['Ej']}_{rpoints[i]['phi_dc']}_{rpoints[i]['Lambdas']}_{rpoints[i]['L1']}_{rpoints[i]['L2']}"
            rpoints[i]["name"] = name
            if name not in qubit_names.keys():
                qubit_names[name] = len(qubit_names)
            rpoints[i]["index"] = qubit_names[name]
            rpoints[i]["truncation"] = self.truncation
            rpoints[i]["base_ex"] = np.pi*6
            rpoints[i]["base_size"] = 2000
        #save the qubit names
        with open("qubit_names.pickle", "wb") as f:
            pickle.dump(qubit_names, f)

        #return these
        return rpoints
        
    def feed_points(self, points):
        if "adaptive" in self.framework:
            
            #points are expected to be a list of dictionaries
            if len(self.sublearners) > 0:
                raise NotImplementedError("Sublearners are not implemented yet")
            for point in points:
                score = point["score"]
                point = np.array([point[k] for k in self.point_keys])
                #use mask
                point = point[self.eval_mask]
                #store these
                if len(self.xs) == 0:
                    self.xs = np.array([point])
                    self.ys = np.array([score])
                else:
                    self.xs = np.vstack((self.xs, point))
                    self.ys = np.append(self.ys, score)
                try:
                    self.learner.tell(point, score)
                except ValueError as e:
                    if "Point already in triangulation" in str(e):
                        print(f"Point {point} already in triangulation")
                    else:
                        raise ValueError(f"ValueError: {e}")
        elif self.framework == "bayesian":
            #points are expected to be a list of dictionaries
            if len(self.sublearners) > 0:
                raise NotImplementedError("Sublearners are not implemented yet")
            for point in points:
                score = point["score"]
                point = np.array([point[k] for k in self.point_keys])
                #use mask
                point = point[self.eval_mask]
                #store these
                if len(self.xs) == 0:
                    self.xs = np.array([point])
                    self.ys = np.array([score])
                else:
                    self.xs = np.vstack((self.xs, point))
                    self.ys = np.append(self.ys, score)
                try:
                    self.learner.tell(point, score,fit=False)
                except ValueError as e:
                    if "Point already in triangulation" in str(e):
                        print(f"Point {point} already in triangulation")
                    else:
                        raise ValueError(f"ValueError: {e}")
        elif self.framework == "random":
            pass
        else:
            raise NotImplementedError(f"Framework {self.framework} is not implemented")
    def convert_Ln_2_decay(self, point):
        #convert the length to decay
        for k in point.keys():
            if k == "L1":
                L1 = point[k]
                c1 = np.array([[0,L1],[0,0]])
                #c1 = np.array([[0,0],[0,0]])
            if k == "L2":
                L2 = point[k]
                c2 = np.array([[L2,0],[0,L2]])
                #c2 = np.array([[0,0],[0,0]])
        point["c_ops"] = [c1, c2]
        return point
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import asyncio
from copy import deepcopy as copy
import qutip as qt
#import qutip_cupy
def get_simple_gate(unitvec,theta,baselen=2):
    sx, sy, sz = qt.sigmax().full(), qt.sigmay().full(), qt.sigmaz().full()
    unitvec = np.array(unitvec)/np.linalg.norm(unitvec)
    unitary = np.eye(2,2,dtype=complex)*np.cos(theta/2)
    unitary += 1j*np.sin(theta/2)*(unitvec[0]*sx + unitvec[1]*sy + unitvec[2]*sz)
    base = np.eye(baselen,baselen,dtype=complex)
    base[:2,:2] = unitary
    #print(base)
    return qt.Qobj(base)
from scipy.optimize import minimize
import sympy as sp



def eval_qubit(gate_params, qubit, t0_samplerate=5, calZ=False):
    
    #global scheme_scores
    #global scheme_scores_raw
    global config
    gate = gate_params.name
    #set params (omega_01, omega_12, etc.)
    #import envelope
    H0, n_opp, phi_opp, t_g = qubit["H0"], qubit["n_opp"], qubit["phi_opp"], qubit["t_g"]
    c_opps = qubit["c_ops"]
    #alpha = qubit["alpha"]
    
    #t_g = gate_params.t_g
    ideal_gate = gate_params.ideal
    #ideal_gate = get_gate_params("ideal")
    loop = []
    if gate_params.known_t0:
        loop = np.array(range(t0_samplerate-1))
    else:
        loop = np.linspace(0, np.pi*((H0[1,1]-H0[0,0]).real**(-1)), t0_samplerate,endpoint=False)
        if np.mean(np.abs(loop)) >t_g*10:
            loop = np.linspace(0, t_g, t0_samplerate)
        #loop = np.zeros(5)
    scores = []
    Z_amps = []
    results_list = []
    for cnt,t0 in enumerate(loop):#for each probing t0
        if gate_params.known_t0: 
            n = t0
            dt0 = gate_params.dt0_amp
            print(f"Testing at n = {n}")
            t0 = None
        else:
            print(f"Testing at t0 = {t0}")
            n = None
            dt0 = None
        gate_dynamics = copy(gate_params).assert_H0(copy(H0))
        #print(n_opp, phi_opp)
        gate_dynamics = copy(gate_dynamics.assert_opps(copy(n_opp), copy(phi_opp)))
        gate_dynamics = copy(gate_dynamics.compile_as_Qobj())
        H0_sim, H_sim = copy(gate_dynamics.transform_2_rotating_frame(t_g,omega_01=copy(H0[1,1]-H0[0,0]),t0=t0, n=n,dt0=dt0))
        #print(H_sim(2).full())
        #H = gate_dynamics.get_QuTiP_compile()
        #either opp-type or state-type solver
        if calZ:
            opp_solvers, start_states = gs.init_sim(H0_sim, H_sim, c_opps, n_opp, phi_opp,initial_state="even",opp_solver=True)
        else:
            solvers, start_states = gs.init_sim(H0_sim, H_sim, c_opps, n_opp, phi_opp,initial_state="even",opp_solver=False)
        if "H_log.txt" in os.listdir("temp") and config["H_log"] and cnt==0: os.remove("temp/H_log.txt")
        if "H_log11.txt" in os.listdir("temp") and config["H_log"] and cnt==0: os.remove("temp/H_log11.txt")
        if "H_log.txt" in os.listdir("temp") and config["H_log"]:
            with open("temp/H_log.txt", "a") as f:
                f.write(f"breakpoint\n")
        if n != None:
            t0 = gate_dynamics.t_0
            t0 = t0.subs(sp.symbols("n"), 2*n)
            t0 = t0.subs(sp.symbols("t_g"), t_g)
            t0 = t0.subs(sp.symbols("omega_{01}"), H0[1,1]-H0[0,0])
            t0 = t0.subs(sp.symbols("dt_0"), dt0)
            t0 = t0.evalf()
        #run solver
        if calZ: #to find opps for Z-opt
            r = gs.simulate(opp_solvers, start_states, t0,1.1*t_g+t0)
        else:#using the state solver and already found Z
            #raise NotImplementedError("Do inverse of Z on initial states")
            if not np.all(gate_params.VZ_amp == 0):
                z_d = gate_params.VZ_amp[0] 
                z_s = gate_params.VZ_amp[1]
                U_d_inv = get_simple_gate([0,0,1], -z_d, baselen=len(start_states[0].full()))
                ss_local = []
                for i in range(len(start_states)):
                    ss = U_d_inv*start_states[i]*U_d_inv.dag()
                    ss_local.append(ss)
                r = gs.simulate(solvers, ss_local, t0,1.1*t_g+t0)#!edit
                U_d = get_simple_gate([0,0,1], z_d, baselen=len(start_states[0].full()))
                U_s = get_simple_gate([0,0,1], z_s, baselen=len(start_states[0].full()))
                for i in range(len(r)):
                    r[i].states[0] = U_d*r[i].states[0]*U_d.dag()
                    r[i].states[-1] = U_s*r[i].states[-1]*U_s.dag()
                    r[i].states[-1] = U_d*r[i].states[-1]*U_d.dag()
            else:
                r = gs.simulate(solvers, start_states, t0,1.1*t_g+t0)
        #store
        for i in range(len(r)):
            r[i].name = f"{qubit['name']}_{gate}_t0_{t0}_i_{i}"
        if calZ: opperators = r
        else: results = r
        if calZ:#optimize Z-amps
            """for i in range(len(results)):
                def score_at_Zamp(Zamp):
                    unitary = get_simple_gate([0,0,1], Zamp, baselen=len(results[i].states[0].full()))
                    res_clone = copy(results[i])
                    res_clone.states[-1] = unitary*res_clone.states[-1]*unitary.dag()
                    score = ge.evaluate([res_clone],ideal_gate,light=True)
                    return -score.real
                #minimize the score
                r = minimize(score_at_Zamp, 0.01, bounds=[(-1,1)])
                val_best = r.x[0]   
                Z_amps.append(val_best)"""
            #instead of the above, minimize the avg fidelity over all results
            a_s = []
            theta_cords = []
            def score_at_Zamp(Zamps):
                Z_double = Zamps[0]
                Z_single = Zamps[1]
                unitary_d = get_simple_gate([0,0,1], Z_double, baselen=len(start_states[0].full()))
                unitary_d_inv = get_simple_gate([0,0,1], -Z_double, baselen=len(start_states[0].full()))
                unitary_s = get_simple_gate([0,0,1], Z_single, baselen=len(start_states[0].full()))
                #print(unitary.full())
                #print(unitary_inv.full())
                #unitary = qt.spre(unitary) * qt.spost(unitary.dag())
                #unitary_inv = qt.spre(unitary_inv) * qt.spost(unitary_inv.dag())
                scores = []
                
                for i in range(len(opperators)):
                    opp = copy(opperators[i])
                    U_d = qt.to_super(unitary_d)
                    U_d_inv = qt.to_super(unitary_d_inv)
                    U_s = qt.to_super(unitary_s)
                    opp_final = U_d @ U_s @ opp @ U_d_inv
                    #opp_final = U @ opp @ U_inv
                    ss = qt.operator_to_vector(start_states[i])
                    fs = opp_final @ ss
                    fs = qt.vector_to_operator(fs)
                    scores.append(ge.evaluate_onstate([fs],[start_states[i]],ideal_gate))

                    """opp = qt.Qobj(1/np.sqrt(2)*(np.eye(2,2)+1j*qt.sigmay().full()))
                    U = unitary*opp*unitary_inv
                    fs = U*start_states[i]*U.dag()
                    scores.append(ge.evaluate_onstate([fs],[start_states[i]],ideal_gate))"""

                    """opp = qt.Qobj(1/np.sqrt(2)*(np.eye(2,2)+1j*qt.sigmay().full()))
                    opp = qt.to_super(opp)
                    U = qt.to_super(unitary)
                    U_inv = qt.to_super(unitary_inv)
                    opp_final = U @ opp @ U_inv
                    ss = qt.operator_to_vector(start_states[i])
                    fs = opp_final @ ss
                    fs = qt.vector_to_operator(fs)
                    scores.append(ge.evaluate_onstate([fs],[start_states[i]],ideal_gate))"""
                    #scores.append(ge.evaluate_onstate([ss3],[start_states[i]],ideal_gate))
                a = np.log10(1-np.mean(scores).real)#!edit
                theta_cords.append(Zamps)
                a_s.append(a)  
                #plot
                if False and np.random.uniform() < 0.1:
                    plt.clf()
                    plt.scatter(np.array(theta_cords)[:,0], np.array(theta_cords)[:,1], c=a_s)
                    plt.colorbar()
                    plt.xlabel("Z_double")
                    plt.ylabel("Z_single")
                    plt.title(f"Fidelity for {qubit['name']} with {gate} at t0 = {t0}")
                    plt.savefig(f"temp/{qubit['name']}_{gate}_t0_{t0}.png")
                    plt.savefig(f"temp/zamps.png")
                    plt.show()
                    plt.close()
                    plt.close('all')
                    plt.clf()
                if a <= -7:
                    raise StopIteration("Target fidelity reached")
                return a
            #minimize the score
            #start by trying 4
            #thetas = [0, np.pi/2, np.pi, 3*np.pi/2]
            #losses = [score_at_Zamp(theta) for theta in thetas]
            #val_best = thetas[np.argmin(losses)]
            try:
                r = minimize(score_at_Zamp, (0,0), method="Nelder-Mead")
                val_best = r.x
            except StopIteration as e:
                print(e)
                indx_best = np.argmin(a_s)
                val_best = theta_cords[indx_best]

            Z_amps.append(val_best)
        if calZ:
            results_list.append(opperators)
        else:
            results_list.append(results)
    if calZ:#pick best Z and apply (across t0's)
        Z_amp_avg = np.mean(Z_amps,axis=0)
        z_d = Z_amp_avg[0]
        z_s = Z_amp_avg[1]
        unit_d = get_simple_gate([0,0,1], z_d, baselen=len(start_states[0].full()))
        unit_d_inv = get_simple_gate([0,0,1], -z_d, baselen=len(start_states[0].full()))
        unit_s = get_simple_gate([0,0,1], z_s, baselen=len(start_states[0].full()))
        #unit = qt.spre(unit) * qt.spost(unit.dag())
        #unit_inv = qt.spre(unit_inv) * qt.spost(unit_inv.dag())
        class ResultPacker:
            def __init__(self, states):
                self.states = states
        for i in range(len(results_list)):
            for j in range(len(results_list[i])):
                opp = copy(results_list[i][j])
                U_d = qt.to_super(copy(unit_d))
                U_d_inv = qt.to_super(copy(unit_d_inv))
                U_s = qt.to_super(copy(unit_s))
                opp_final = U_d @ U_s @ opp @ U_d_inv
                ss = qt.operator_to_vector(copy(start_states[j]))
                fs = opp_final @ ss
                fs = qt.vector_to_operator(fs)
                results_list[i][j] = ResultPacker([fs])
                scores.append(ge.evaluate_onstate([fs],[start_states[j]],ideal_gate))
                """ss = qt.operator_to_vector(start_states[j])
                ss1 = unit_inv*ss
                ss2 = results_list[i][j]*ss1
                ss3 = unit*ss2
                ss3 = qt.vector_to_operator(ss3)
                results_list[i][j] = ResultPacker([ss3])"""
                #results_list[i][j].states[-1] = unit*results_list[i][j].states[-1]*unit.dag()
        gate_params.VZ_amp = Z_amp_avg
        gate_params.VZ_amp_buffer = []
        if config["bloch"] and False:
            z_d = gate_params.VZ_amp[0]
            z_s = gate_params.VZ_amp[1]
            solvers_loc, start_states_loc = gs.init_sim(H0_sim, H_sim, c_opps, n_opp, phi_opp,initial_state="even",opp_solver=False)
            old = copy(start_states_loc)
            U_d_inv = get_simple_gate([0,0,1], -z_d, baselen=len(start_states[0].full()))
            for i in range(len(start_states)):
                start_states_loc[i] = U_d_inv*start_states_loc[i]*U_d_inv.dag()
                #print((start_states_loc[i]-old[i]).full())
            r = gs.simulate(solvers_loc, start_states_loc, t0,1.1*t_g+t0)
            U_s = get_simple_gate([0,0,1], z_s, baselen=len(r[0].states[0].full()))
            U_d = get_simple_gate([0,0,1], z_d, baselen=len(r[0].states[0].full()))
            for i in range(len(r)):
                #for j in range(len(r[i].states)):
                #    r[i].states[j] = U*r[i].states[j]*U.dag()
                r[i].states[0] = U_d*r[i].states[0]*U_d.dag()#making explicit the einitial transformation for visual pruporses
                r[i].states[-1] = U_s*r[i].states[-1]*U_s.dag()#creating the last transformation for fidelity purposes
                r[i].states[-1] = U_d*r[i].states[-1]*U_d.dag()#creating the last transformation for fidelity purposes
            for i in range(len(r)):
                ge.evaluate([r[i]],ideal_gate)
            pass
    """elif gate_params.VZ_amp != 0:
        
        unit = get_simple_gate([0,0,1], gate_params.VZ_amp, baselen=len(results[0].states[0].full()))
        for i in range(len(results_list)):
            for j in range(len(results_list[i])):
                results_list[i][j].states[-1] = unit*results_list[i][j].states[-1]*unit.dag()"""
    for i in range(len(results_list)):
        results = results_list[i]
        if len(results[0].states) > 1:#wether only one state or full states is present
            score = ge.evaluate(results,ideal_gate)
        else:
            score = ge.evaluate_onstate(results,start_states,ideal_gate)
        scores.append(score)
        if config["H_log"]:
            with open("temp/H_log.txt", "r") as f:
                lines = f.readlines()
                #find lines with "breakpoint"
                bps = [i for i in range(len(lines)) if "breakpoint" in lines[i]]
                lines = [l for i,l in enumerate(lines) if i not in bps]
                lines = np.array([[float(line.strip().split(";")[0]),complex(line.strip().split(";")[1])] for line in lines], dtype=complex)
                X = lines.T[0]
                Y = lines.T[1]
                plt.clf()
                plt.scatter(X, Y.real, s=0.1)
                plt.scatter(X, Y.imag, s=0.1)

                #integrate Y between the first two breakpoints
                y_int = np.trapz( Y[:bps[0]], x=X[:bps[0]])
                print(f"y contribution avg value: {y_int.imag/y_int.real}")
                #find the highest index of y that is > 0
                nonz = ~np.isclose(Y.real,0)
                #last true value
                last_true = np.where(nonz)[0][-1]
                maxX = X[last_true]
                plt.xlim(np.min(X), maxX)
                plt.xlabel("Time")
                plt.ylabel("H(t)")
                plt.title(f"H(t) for {qubit['name']} with {gate}")
                plt.savefig(f"temp/H_log_{qubit['name']}_{gate}_t0_{t0}.png")
                plt.savefig(f"temp/H_log.png")
                plt.clf()
                plt.close()

    
        print(f"Score for {qubit['name']} with {gate}: {score}")
    #print(score)
    #if qubit["name"] not in scheme_scores.keys(): scheme_scores[qubit["name"]] = {}
    if not np.isclose(score,score.real).all():
        print("Score has complex values")
    s = np.mean(scores).real
    #scheme_scores[qubit["name"]][gate] = s
    #also save this in scheme_scores_raw

    dp = {  
        "Ec": qubit["Ec"],
        "El": qubit["El"],  
        "Ej": qubit["Ej"],
        "phi_dc": qubit["phi_dc"],
        "t_g": qubit["t_g"],
        "Lambdas": qubit["Lambdas"],
        "alpha_actual": qubit["alpha_actual"],
        "alpha_target": qubit["alpha_target"],
        "omega_01_actual": qubit["omega_01_actual"],
        "omega_01_target": qubit["omega_01_target"],
        "score": s,
        "gate": gate,
        "L1": qubit["L1"],
        "L2": qubit["L2"],
    }
    #scheme_scores_raw.append(dp)
    return s,dp, gate_params





def single_param(gate_params, qubit, gate, do_VZ=False):
    for key in gate_params.is_calibrated.keys():
        if "VZ" in key:
            continue#Z routine is associated with omega
        valname = key
        Z_now = False
        if "Omega" in key:
            Z_now = True
        if gate_params.is_calibrated[key] == False:
            """gate_in_matrix = True
            qb_in_matrix = True
            #first check if (A) calibration matrix exists, and (B) if it has an entry for this gate/qubit
            gate_name = gate_params.name
            matrixname = f"temp/calibration_matrix_val={key}.pickle"
            if Z_now: matrixname2 = f"temp/calibration_matrix_val=VZ.pickle"
            if os.path.exists(matrixname):
                while os.path.exists(matrixname.replace(".pickle", "")):#a read protection file
                    time.sleep(0.1)
                with open(matrixname, "rb") as f:
                    matrix = pickle.load(f)
                if Z_now :
                    with open(matrixname2, "rb") as f:
                        matrix2 = pickle.load(f)
                if gate_name in matrix.keys():
                    #check if the qubit is in the matrix
                    if qubit["index"] in matrix[gate_name].keys():
                        #get the value
                        val = matrix[gate_name][qubit["index"]]
                        if Z_now:
                            val2 = matrix2[gate_name][qubit["index"]]
                        #assert the value
                        #gate_params.assert_calibration(key, val)
                        exec(f"gate_params.{valname}_amp = val")
                        if Z_now:
                            exec(f"gate_params.VZ_amp = val2")
                    else:
                        qb_in_matrix = False
                else:
                    gate_in_matrix = False
                    qb_in_matrix = False
            else:
                matrix = {}
                if Z_now: matrix2 = {}
                gate_in_matrix = False
                qb_in_matrix = False
            if not gate_in_matrix:
                #create a new matrix
                matrix[gate_name] = {}
                if Z_now: matrix2[gate_name] = {}
                if not qb_in_matrix:
                    #create a new entry for the qubit
                    matrix[gate_name][qubit["index"]] = 0
                    if Z_now: matrix2[gate_name][qubit["index"]] = 0
            #next actually do the calibration"""
            if True:#not gate_in_matrix or not qb_in_matrix:
                #create a new matrix
                
                #do the calibration
                H0, n_opp, phi_opp, c_ops, t_g, base_ex, base_size, Ec, El, Ej = qbi.init_qubit(qubit["Ec"], qubit["El"], qubit["Ej"], qubit["phi_dc"],qubit['omega_01_target'],qubit['alpha_target'], qubit["c_ops"], qubit["Lambdas"],truncation=qubit["truncation"], base_ex=qubit["base_ex"], base_size=qubit["base_size"],Ec_IC=qubit["Ec_IC"],El_IC=qubit["El_IC"],Ej_IC=qubit["Ej_IC"])
                qubit["base_ex"] = base_ex
                qubit["base_size"] = base_size
                if H0 is None:
                    print("H0 is None")
                    return None
                #! is it diagonalized?
                qubit["H0"] = H0
                if len(H0) <= 2: qubit['alpha_actual'] = np.inf
                else: qubit['alpha_actual'] = H0[2,2]-H0[1,1] - (H0[1,1]-H0[0,0])
                qubit["omega_01_actual"] = H0[1,1]-H0[0,0]
                qubit["n_opp"] = n_opp
                qubit["phi_opp"] = phi_opp
                qubit["c_ops"] = c_ops
                qubit["t_g"] = t_g
                qubit["Ej_actual"] = Ej
                qubit["El_actual"] = El
                qubit["Ec_actual"] = Ec
                dummyfunc = lambda x: None
                #loss_per_interval = lambda xs,ys: np.diff(xs)*(np.mean(ys)**(-1))#!I have no clue what y is at this point, but this seem to give the correct behavior
                if "Omega" in valname:
                    bnds = (0, 10/t_g)
                elif "dt0" in valname:
                    initial = gate_params.dt0_amp.subs(sp.symbols("t_g"), t_g)
                    bnds = (initial-0.5*np.abs(initial), initial)
                #loss_per_interval = lambda xs,ys: (np.power(np.diff(xs)/(bnds[1]-bnds[0]),-1))*(1.05-np.mean(ys))*np.max(np.array(xs)+1)*(np.min(np.append(-np.diff(ys).flatten(),0))+0.5)
                #loss_per_interval = lambda xs,ys: np.piecewise(
                #    (xs,ys),
                #    (ys[1]-ys[0] < 0 & np.isclose(xs[0],0),True),
                #    (-np.infty,lambda xs,ys: (np.power(np.diff(xs)/(bnds[1]-bnds[0]),-1))*(1.05-np.mean(ys))*np.mean(np.array(xs)+1))
                #)
                if "Omega" in valname:
                    def loss_per_interval(xs,ys):
                        if np.isclose(xs[0],0) and ys[1]-ys[0] < 0:
                            return -np.inf, True
                        else:
                            return (np.power(np.diff(xs)/(bnds[1]-bnds[0]),-1))*(1.01-np.mean(ys))*np.mean(np.array(xs)+1), False
                else:
                    def loss_per_interval(xs,ys):
                         return (np.power(np.diff(xs)/(bnds[1]-bnds[0]),-1))*(1.01-np.mean(ys)), False
                #Sampler = adaptive.Learner1D(dummyfunc, (0, 10), loss_per_interval=loss_per_interval)
                Sampler = Adaptive_1D_Custom(dummyfunc, bnds, loss_per_interval=loss_per_interval)
                for i in range(14):#!temp
                    point = Sampler.ask(1)
                    exec(f"gate_params.{valname}_amp = point[0][0]")
                    score, datapoint, gate_params = eval_qubit(gate_params, qubit, t0_samplerate=3, calZ=False)
                    Sampler.tell(point[0][0], score)
                    x,y = [],[]
                    if (i%5 == 0 and i>10) or i == 9 or i == 13:
                        #find the peak, fit a polynomial, and eval the peak
                        data = Sampler.to_numpy().astype(float)
                        val_best = np.max(data.T[1])
                        inx_best = np.argmax(data.T[1])
                        few_best = np.argsort(np.abs(data.T[0]-data.T[0][inx_best]))[:3]
                        #fit a polynomial
                        coeffs = np.polyfit(data.T[0][few_best], data.T[1][few_best], 4)
                        #get the peak
                        x = np.linspace(data.T[0][few_best].min(), data.T[0][few_best].max(), 100)
                        y = np.polyval(coeffs, x)
                        
                        peak_val = x[np.argmax(y)]
                        if peak_val > 0:
                            exec(f"gate_params.{valname}_amp = peak_val")
                            score, datapoint, _ = eval_qubit(gate_params, qubit, t0_samplerate=3, calZ=False)
                            Sampler.tell(peak_val, score)

                    data = Sampler.to_numpy()
                    points = data.T[0]
                    scores = data.T[1]
                    if 1>0:
                        plt.plot(x, 1-np.array(y))
                        plt.scatter(points, 1-scores)
                        plt.xlabel("Value")
                        plt.ylabel("Score")
                        plt.title(f"Calibration for {qubit['name']} with {gate}")
                        plt.savefig(f"temp/calibration_{qubit['name']}_{gate}.png")
                        plt.savefig(f"temp/calibration.png")
                        plt.clf()
                        plt.close()
                data = Sampler.to_numpy()
                points = data.T[0]
                scores = data.T[1]
                indx_best = np.argmax(scores)
                val = points[indx_best]
                exec(f"gate_params.{valname}_amp = val")
                if do_VZ:
                    score, datapoint, gate_params = eval_qubit(gate_params, qubit, t0_samplerate=3, calZ=True)
                #put in matrix
                matrix[gate_name][qubit["index"]] = val
                if Z_now:
                    matrix2[gate_name][qubit["index"]] = gate_params.VZ_amp
            #save the matrix
            while os.path.exists(matrixname.replace(".pickle", "")):#a read protection file
                time.sleep(0.1)
            Path(matrixname.replace(".pickle", "")).touch()
            with open(matrixname.replace('.', '_temp.'), "wb") as f:
                pickle.dump(matrix, f)
            os.system(f"mv {matrixname} {matrixname.replace('.', '_old.')}")
            os.system(f"mv {matrixname.replace('.', '_temp.')} {matrixname}")
            os.system(f"rm {matrixname.replace('.', '_old.')}")
            if Z_now:
                with open(matrixname2, "wb") as f:
                    pickle.dump(matrix2, f)
            try:
                os.remove(matrixname.replace(".pickle", ""))
            except FileNotFoundError:
                pass
    return qubit, gate_params

import matplotlib
cnt = 0
def two_param(gate_params, qubit, gate, do_VZ=False):
    """for key in gate_params.is_calibrated.keys():
        if "VZ" in key:
            continue#Z routine is associated with omega"""
    keys = [k for k in gate_params.is_calibrated.keys() if "VZ" not in k]
    #valname = key
    #Z_now = False
    #if "Omega" in key:
    #    Z_now = True
    #!skipping matrix stuff
    #if gate_params.is_calibrated[key] == False:
    #    gate_in_matrix = True
    #    qb_in_matrix = True
    #    #first check if (A) calibration matrix exists, and (B) if it has an entry for this gate/qubit
    #    gate_name = gate_params.name
    #    matrixname = f"temp/calibration_matrix_val={key}.pickle"
    #    if Z_now: matrixname2 = f"temp/calibration_matrix_val=VZ.pickle"
    #    if os.path.exists(matrixname):
    #        while os.path.exists(matrixname.replace(".pickle", "")):#a read protection file
    #            time.sleep(0.1)
    #        with open(matrixname, "rb") as f:
    #            matrix = pickle.load(f)
    #        if Z_now :
    #            with open(matrixname2, "rb") as f:
    #                matrix2 = pickle.load(f)
    #        if gate_name in matrix.keys():
    #            #check if the qubit is in the matrix
    #            if qubit["index"] in matrix[gate_name].keys():
    #                #get the value
    #                val = matrix[gate_name][qubit["index"]]
    #                if Z_now:
    #                    val2 = matrix2[gate_name][qubit["index"]]
    #                #assert the value
    #                #gate_params.assert_calibration(key, val)
    ##                exec(f"gate_params.{valname}_amp = val")
    #               if Z_now:
    #                    exec(f"gate_params.VZ_amp = val2")
    #            else:
    #                qb_in_matrix = False
    #        else:
    #            gate_in_matrix = False
    #            qb_in_matrix = False
    #    else:
    #        matrix = {}
    #        if Z_now: matrix2 = {}
    #        gate_in_matrix = False
    #        qb_in_matrix = False
    #    if not gate_in_matrix:
    #        #create a new matrix
    #        matrix[gate_name] = {}
    #        if Z_now: matrix2[gate_name] = {}
    ##        if not qb_in_matrix:
    # #           #create a new entry for the qubit
    #            matrix[gate_name][qubit["index"]] = 0
    #            if Z_now: matrix2[gate_name][qubit["index"]] = 0
    #    #next actually do the calibration
    #    if not gate_in_matrix or not qb_in_matrix:
            #create a new matrix
            
    #do the calibration
    H0, n_opp, phi_opp, c_ops, t_g, base_ex, base_size, Ec, El, Ej = qbi.init_qubit(qubit["Ec"], qubit["El"], qubit["Ej"], qubit["phi_dc"],qubit['omega_01_target'],qubit['alpha_target'], qubit["c_ops"], qubit["Lambdas"], truncation=qubit["truncation"], base_ex=qubit["base_ex"], base_size=qubit["base_size"],Ec_IC=qubit["Ec_IC"],El_IC=qubit["El_IC"],Ej_IC=qubit["Ej_IC"])
    qubit["base_ex"] = base_ex
    qubit["base_size"] = base_size
    if H0 is None:
        print("H0 is None")
        return None
    #! is it diagonalized?
    qubit["H0"] = H0
    if len(H0) <= 2: qubit['alpha_actual'] = np.inf
    else: qubit['alpha_actual'] = H0[2,2]-H0[1,1] - (H0[1,1]-H0[0,0])
    qubit["omega_01_actual"] = H0[1,1]-H0[0,0]
    qubit["n_opp"] = n_opp
    qubit["phi_opp"] = phi_opp
    qubit["c_ops"] = c_ops
    qubit["t_g"] = t_g
    qubit["Lambdas"] = qubit["Lambdas"]
    qubit["Ec_actual"] = Ec
    qubit["El_actual"] = El
    qubit["Ej_actual"] = Ej
    #dummyfunc = lambda x: None
    #loss_per_interval = lambda xs,ys: np.diff(xs)*(np.mean(ys)**(-1))#!I have no clue what y is at this point, but this seem to give the correct behavior
    #bnds = (0, 2)
    #loss_per_interval = lambda xs,ys: (np.power(np.diff(xs)/(bnds[1]-bnds[0]),-1))*(1.05-np.mean(ys))*np.max(np.array(xs)+1)*(np.min(np.append(-np.diff(ys).flatten(),0))+0.5)
    #loss_per_interval = lambda xs,ys: np.piecewise(
    #    (xs,ys),
    #    (ys[1]-ys[0] < 0 & np.isclose(xs[0],0),True),
    #    (-np.infty,lambda xs,ys: (np.power(np.diff(xs)/(bnds[1]-bnds[0]),-1))*(1.05-np.mean(ys))*np.mean(np.array(xs)+1))
    #)
    #keys are found
    #initial values
    gate_params
    initial_values = np.ones(len(keys), dtype=float)
    bounds = []
    scaling = []
    periodicity = []
    for i, key in enumerate(keys):
        initial = eval(f"gate_params.{key}_amp")
        #gate_params[f"initial_{key}"]
        #if its a sympy type, sub relevant values
        if isinstance(initial, sp.Basic):
            initial = initial.subs(sp.Symbol('dt_0'), gate_params.dt0_amp)
            initial = initial.subs(sp.Symbol('t_g'), t_g)
            initial = initial.subs(sp.Symbol('omega_{01}'), H0[1,1]-H0[0,0])
        val = complex(initial)
        initial_values[i] = float(val.real)
        if "Omega" in key:
            bounds.append((0.75*val.real,1.25*val.real))
            periodicity.append(None)
        elif "dt0" in key:
            periodicity.append(np.pi/qubit["omega_01_actual"])
            bounds.append((0, periodicity[-1]))
        else:
            raise NotImplementedError(f"Key {key} not implemented")
        scaling.append(np.abs(bounds[-1][1]-bounds[-1][0]))
    gates = []
    scores = []
    cord = []
    #cnt = 0
    def loss_func(vals):
        global cnt
        #vals = np.array(vals)*np.power(scaling, 1)
        local_gate = copy(gate_params)
        for i, key in enumerate(keys):
            if periodicity[i] != None:
                vals[i] = np.mod(vals[i], periodicity[i])
            exec(f"local_gate.{key}_amp = vals[i]")
        score, datapoint, local_gate = eval_qubit(local_gate, qubit, t0_samplerate=3, calZ=True)
        gates.append(local_gate)
        scores.append(-score)
        cord.append(vals)
        if 1 < 0:
            plt.clf()
            cmap = plt.get_cmap("viridis")
            plt.scatter(np.array(cord).T[0], np.array(cord).T[1], c=1+np.array(scores), cmap=cmap, norm=matplotlib.colors.LogNorm())
            #fit a polynomial landscape to the data (cord 1+2, score)
            #coeff, r, rank, s = np.linalg.lstsq(np.array(cord), np.array(scores))
            #func = lambda x,y: coeff[3]*x**2 + coeff[6]*y**2 
            plt.colorbar()
            plt.xlabel(keys[0])
            plt.ylabel(keys[1])
            plt.title(f"Calibration for {qubit['name']} with {gate}")
            plt.savefig(f"temp/optimize_{qubit['name']}_{gate}.png")
            plt.savefig(f"temp/optimize.png")
            plt.clf()
            plt.close("all")
        cnt += 1
        print(f"Count: {cnt}")
        if 1-score <= 2e-6:
            raise StopIteration
        return np.log10(1-score)
    #do a simple scipy minimize
    try:
        #_ = minimize(loss_func, initial_values, options={"maxiter": 20},method="Nelder-Mead")
        #i_vars = np.array(initial_values)*np.power(scaling, -1)
        #bnds = [np.array(bounds[i])*np.power(scaling[i], -1) for i in range(len(bounds))]
        #r = basinhopping(loss_func, i_vars, niter=10, T=3, stepsize=1, minimizer_kwargs={"method": "Nelder-Mead", "options": {"maxiter": 6}, "bounds": bnds},disp=True)
        #use an adaptive sampler wrapper for as a better basin-hopping
        def basinfunction(x, sampler,iter=10):
            lastlen= len(cord)
            r = minimize(loss_func, x, method="Nelder-Mead", options={"maxiter": iter},bounds=bounds)
            #feed points to sampler
            for i in range(lastlen, len(cord)):
                sampler.tell(cord[i], scores[i])
            return sampler
        dummy = lambda x: None
        sampler = adaptive.Learner2D(dummy, bounds, loss_per_triangle=adaptive.learner.learner2D.areas)
        basinfunction(initial_values, sampler,iter=30)
        #basinfunction((0,i_vars[1]), sampler)
        #basinfunction((bnds[0][1],i_vars[1]), sampler)
        # sampler.bounds_are_done = True
        def distance(pnt,cord):
            dists = pnt - np.array(cord)
            dists[:,0] /= np.abs(bounds[0][1]-bounds[0][0])
            dists[:,1] /= np.abs(bounds[1][1]-bounds[1][0])
            dists = np.linalg.norm(dists, axis=1)
            return np.min(dists)
        for i in range(10):
            iter = 13
            if i <= 1: iter = 1
            #pnt = sampler.ask(1)
            pnt = (np.random.uniform(bounds[0][0], bounds[0][1]), np.random.uniform(bounds[1][0], bounds[1][1]))
            lim = 0.5
            counter = 0
            while distance(pnt,cord)<lim:
                pnt = (np.random.uniform(bounds[0][0], bounds[0][1]), np.random.uniform(bounds[1][0], bounds[1][1]))
                counter += 1
                if counter > 100:
                    lim *= 0.8
                    counter = 0
            sampler = basinfunction(pnt, sampler, iter=iter)
        best = np.argmin(scores)
        best_val = np.array(cord)[best]
        best_score = scores[best]
        _ = minimize(loss_func, best_val, options={"maxiter": 30},method="Nelder-Mead")
        best = np.argmin(scores)
        best_val = np.array(cord)[best]
        best_score = scores[best]
    except StopIteration:
        print("Target fidelity reached")

    """final = result.x
    for i, key in enumerate(keys):
        exec(f"gate_params.{key}_amp = final[i]")"""
    gate_params = gates[np.argmin(scores)]
    """if do_VZ:
        _, _, gate_params = eval_qubit(gate_params, qubit, t0_samplerate=3, calZ=True)"""
    
    """
    data = Sampler.to_numpy()
    points = data.T[0]
    scores = data.T[1]
    indx_best = np.argmax(scores)
    val = points[indx_best]
    exec(f"gate_params.{valname}_amp = val")
    #put in matrix
    matrix[gate_name][qubit["index"]] = val
    if Z_now:
        matrix2[gate_name][qubit["index"]] = gate_params.VZ_amp
    #save the matrix
    while os.path.exists(matrixname.replace(".pickle", "")):#a read protection file
        time.sleep(0.1)
    Path(matrixname.replace(".pickle", "")).touch()
    with open(matrixname.replace('.', '_temp.'), "wb") as f:
        pickle.dump(matrix, f)
    os.system(f"mv {matrixname} {matrixname.replace('.', '_old.')}")
    os.system(f"mv {matrixname.replace('.', '_temp.')} {matrixname}")
    os.system(f"rm {matrixname.replace('.', '_old.')}")
    if Z_now:
        with open(matrixname2, "wb") as f:
            pickle.dump(matrix2, f)
    try:
        os.remove(matrixname.replace(".pickle", ""))
    except FileNotFoundError:
        pass"""
    return qubit, gate_params

class Adaptive_1D_Custom():
    def __init__(self, function, bounds, loss_per_interval=None):
        self.function = function
        self.bounds = bounds
        self.loss_per_interval = loss_per_interval
        self.xs = []
        self.ys = []
    def ask(self, N=1):
        if len(self.xs) > 1:
            self.xs,indx = np.unique(self.xs, return_index=True)
            self.ys = np.array(self.ys)[indx]
            losses = np.zeros(len(self.xs)-1)
            for i in range(len(self.xs)-1):
                r = self.loss_per_interval((self.xs[i], self.xs[i+1]), (self.ys[i], self.ys[i+1]))
                #losses[i] = self.loss_per_interval((self.xs[i], self.xs[i+1]), (self.ys[i], self.ys[i+1]))
                if len(r) == 2:
                    losses[i] = r[0]
                    if r[1]:
                        #if this is the case, the bounds are too high. Truncate to this x value
                        self.bounds = (self.bounds[0], self.xs[i+1])
                        #any point over this; remove
                        mask = self.xs > self.xs[i+1]
                        self.xs = self.xs[~mask]
                        self.ys = self.ys[~mask]
                        break
                else:
                    losses[i] = r
            #get the index of the minimum loss
            index = np.argmin(losses)
            #get the point in between
            point = (self.xs[index] + self.xs[index+1]) / 2
            return [[point]]
        else:
            #pick from bounds
            for b in self.bounds:
                if not b in self.xs:
                    return [[b]]         
    def tell(self, x, y):
        if len(self.xs) > 0:
            self.xs = np.append(self.xs, x)
            self.ys = np.append(self.ys, y)
        else:
            self.xs = np.array([x])
            self.ys = np.array([y])
    def to_numpy(self):
        #convert to numpy array
        xs = np.array(self.xs)
        ys = np.array(self.ys)
        return np.array([xs, ys]).T
    def get_smarter(self,N=1):
        point = self.ask(N)
        score = self.function(point[0][0])
        self.tell(point[0][0], score)

def do_test(qubit,gate_params):
    H0, n_opp, phi_opp, c_ops, t_g, base_ex, base_size, Ec, El, Ej = qbi.init_qubit(qubit["Ec_actual"], qubit["El_actual"], qubit["Ej_actual"], qubit["phi_dc"],qubit['omega_01_target'],qubit['alpha_target'], qubit["c_ops"], qubit["Lambdas"],truncation=qubit["truncation"], base_ex=qubit["base_ex"], base_size=qubit["base_size"],optimizebasis=False,Ec_IC=qubit["Ec_IC"],El_IC=qubit["El_IC"],Ej_IC=qubit["Ej_IC"])
    qubit["base_ex"] = base_ex
    qubit["base_size"] = base_size
    if H0 is None:
        raise ValueError("H0 is None")
    #! is it diagonalized?
    scores, points = [], []
    qubit["H0"] = H0
    if len(H0) <= 2: qubit['alpha_actual'] = np.inf
    else: qubit['alpha_actual'] = H0[2,2]-H0[1,1] - (H0[1,1]-H0[0,0])
    qubit["omega_01_actual"] = H0[1,1]-H0[0,0]
    qubit["n_opp"] = n_opp
    qubit["phi_opp"] = phi_opp
    qubit["c_ops"] = c_ops
    qubit["t_g"] = t_g
    qubit["Lambdas"] = qubit["Lambdas"]
    #for gate in gate_names_2_eval:
    #    #gate_params = get_gate_params(gate)
    score, datapoint, _ = eval_qubit(gate_params, qubit, t0_samplerate=10)
    scores.append(score)
    points.append(datapoint)
    return scores, points
def calib_gate(args):
    qubit, gate_params = args
    #do the calibration
    #get the gate params
    gate = gate_params.name
    #gate_params = get_gate_params(gate)
    #check keys to calibrate
    do_VZ = False
    if "VZ" in gate_params.is_calibrated.keys():
        do_VZ = True
    keys_wo_VZ = [key for key in gate_params.is_calibrated.keys() if "VZ" not in key]
    if len(keys_wo_VZ) == 1:#singel variable calibration
        qubit, gate_params = single_param(gate_params, qubit, gate, do_VZ)
    elif len(keys_wo_VZ) == 2:
        qubit, gate_params = two_param(gate_params, qubit, gate, do_VZ)
    elif len(keys_wo_VZ) == 0:
        t1 = hasattr(gate_params,"lambda_eq")
        t2 = sp.Symbol("omega_{01}") in gate_params.Omega_eq.free_symbols
        t3 = "t_g" not in qubit.keys()
        if t1 or t2 or t3:
            H0, n_opp, phi_opp, c_ops, t_g, base_ex, base_size, Ec, El, Ej = qbi.init_qubit(qubit["Ec"], qubit["El"], qubit["Ej"], qubit["phi_dc"],qubit['omega_01_target'],qubit['alpha_target'], qubit["c_ops"], qubit["Lambdas"],truncation=qubit["truncation"], base_ex=qubit["base_ex"], base_size=qubit["base_size"],Ec_IC=qubit["Ec_IC"],El_IC=qubit["El_IC"],Ej_IC=qubit["Ej_IC"])
            qubit["Ej_actual"] = Ej
            qubit["Ec_actual"] = Ec
            qubit["El_actual"] = El
            qubit["base_ex"] = base_ex
            qubit["base_size"] = base_size
            qubit["omega_01_actual"] = H0[1,1]-H0[0,0]
            if len(H0) <= 2: qubit['alpha_actual'] = np.inf
            else:            qubit["alpha_actual"] = H0[2,2]-H0[1,1] - (H0[1,1]-H0[0,0])
        if t3: t_g = qubit["Lambdas"]/(H0[1,1]-H0[0,0])*2*np.pi
        else: t_g = qubit["t_g"]
        val = gate_params.Omega_eq.subs(sp.Symbol("t_g"),t_g)
        exec(f"gate_params.Omega_amp = val")
        #one for lambda, one for wether omega_{01} is in Omega_eq
        
        
        if t1 or t2:
            qubit["base_ex"] = base_ex
            qubit["base_size"] = base_size
            omega_01 = np.real(H0[1,1]- H0[0,0])
        if t1:
            val = gate_params.lambda_eq.subs(sp.Symbol("omega_{01}"),omega_01)
            exec(f"gate_params.lambda_amp = val")
        if t2:
            val = gate_params.Omega_eq.subs(sp.Symbol("omega_{01}"),omega_01).subs(sp.Symbol("t_g"),qubit["t_g"])
            exec(f"gate_params.Omega_amp = val")
    return qubit, gate_params

from numpy.linalg import lstsq
def get_opp_from_densities(rho1_set,rho2_set):
    """
    Solves rho2 = K * rho1 * K† for K over sets of rho1 and rho2.
    Assumes 2x2 density matrices.
    Returns least-squares approximate K.
    """
    A_list = []
    b_list = []
    
    for rho1, rho2 in zip(rho1_set, rho2_set):
        # Vectorize rho1 and rho2
        a = rho1.flatten()
        b = rho2.flatten()
        
        # Build Kronecker product matrix for this pair
        M = np.kron(a.conj().T, np.eye(2))
        
        A_list.append(M)
        b_list.append(b)
    
    # Stack matrices
    A = np.vstack(A_list)
    b = np.concatenate(b_list)
    
    # Solve least-squares for vec(K)
    K_vec, residuals, rank, s = lstsq(A, b, rcond=None)
    
    # Reshape back to 2x2
    K = K_vec.reshape(2, 2)
    
    return K