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
        xs = [[dp["Ec"], dp["El"], dp["Ej"], dp['omega_01'], dp['alpha'], dp["phi_dc"],dp["t_g"],dp["L1"],dp["L2"]] for dp in known_dps]
        point_keys = ["Ec", "El", "Ej", "omega_01", "alpha", "phi_dc", "t_g", "L1", "L2"]
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
        t_g_index = self.point_keys.index("t_g")
        for i in list(range(t_g_index)):
            if self.fixed[i] is not None:
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
            name = f"qubit_{rpoints[i]['Ec']}_{rpoints[i]['El']}_{rpoints[i]['Ej']}_{rpoints[i]['phi_dc']}_{rpoints[i]['t_g']}_{rpoints[i]['L1']}_{rpoints[i]['L2']}"
            rpoints[i]["name"] = name
            if name not in qubit_names.keys():
                qubit_names[name] = len(qubit_names)
            rpoints[i]["index"] = qubit_names[name]
            rpoints[i]["truncation"] = self.truncation
            rpoints[i]["base_ex"] = np.pi*4
            rpoints[i]["base_size"] = 1000
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
import asyncio
from copy import deepcopy as copy
import qutip as qt
import qutip_cupy
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
    alpha = qubit["alpha"]
    
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
    for cnt,t0 in enumerate(loop):
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
        print(H_sim(2).full())
        #H = gate_dynamics.get_QuTiP_compile()
        solvers, start_states = gs.init_sim(H0_sim, H_sim, c_opps, n_opp, phi_opp,initial_state="even")
        if "H_log.txt" in os.listdir("temp") and config["H_log"] and cnt==0: os.remove("temp/H_log.txt")
        if "H_log11.txt" in os.listdir("temp") and config["H_log"] and cnt==0: os.remove("temp/H_log11.txt")
        if "H_log.txt" in os.listdir("temp") and config["H_log"]:
            with open("temp/H_log.txt", "a") as f:
                f.write(f"breakpoint\n")
        if n != None:
            t0 = gate_dynamics.t_0
            t0 = t0.subs(sp.symbols("n"), n)
            t0 = t0.subs(sp.symbols("t_g"), t_g)
            t0 = t0.subs(sp.symbols("omega_{01}"), H0[1,1]-H0[0,0])
            t0 = t0.subs(sp.symbols("dt_0"), dt0)
            t0 = t0.evalf()
        results = gs.simulate(solvers, start_states, t0,1.1*t_g+t0)
        for i in range(len(results)):
            results[i].solver = f"{qubit['name']}_{gate}_t0_{t0}_i_{i}"
        if calZ:
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
            def score_at_Zamp(Zamp):
                unitary = get_simple_gate([0,0,1], Zamp, baselen=len(results[0].states[0].full()))
                score = 0
                for i in range(len(results)):
                    res_clone = copy(results[i])
                    res_clone.states[-1] = unitary*res_clone.states[-1]*unitary.dag()
                    score += ge.evaluate([res_clone],ideal_gate,light=True)
                return -score.real/len(results)
            #minimize the score
            r = minimize(score_at_Zamp, 0.00, bounds=[(-np.pi,np.pi)])
            val_best = r.x[0]
            Z_amps.append(val_best)
        results_list.append(results)
    if calZ:
        Z_amp_avg = np.median(Z_amps)
        unit = get_simple_gate([0,0,1], Z_amp_avg, baselen=len(results[0].states[0].full()))
        for i in range(len(results_list)):
            for j in range(len(results_list[i])):
                results_list[i][j].states[-1] = unit*results_list[i][j].states[-1]*unit.dag()
        gate_params.VZ_amp = Z_amp_avg
        gate_params.VZ_amp_buffer = []
    elif gate_params.VZ_amp != 0:
        unit = get_simple_gate([0,0,1], gate_params.VZ_amp, baselen=len(results[0].states[0].full()))
        for i in range(len(results_list)):
            for j in range(len(results_list[i])):
                results_list[i][j].states[-1] = unit*results_list[i][j].states[-1]*unit.dag()
    for i in range(len(results_list)):
        results = results_list[i]
        score = ge.evaluate(results,ideal_gate)
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
        "alpha": qubit["alpha"],
        "omega_01": qubit["omega_01"],
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
                H0, n_opp, phi_opp, c_ops, t_g, base_ex, base_size = qbi.init_qubit(qubit["Ec"], qubit["El"], qubit["Ej"], qubit["phi_dc"],qubit['omega_01'],qubit['alpha'], qubit["c_ops"], qubit["t_g"],truncation=qubit["truncation"], base_ex=qubit["base_ex"], base_size=qubit["base_size"])
                qubit["base_ex"] = base_ex
                qubit["base_size"] = base_size
                if H0 is None:
                    print("H0 is None")
                    return None
                #! is it diagonalized?
                qubit["H0"] = H0
                if len(H0) <= 2: qubit['alpha'] = np.inf
                else: qubit['alpha'] = H0[2,2]-H0[1,1] - (H0[1,1]-H0[0,0])
                qubit["omega_01"] = H0[1,1]-H0[0,0]
                qubit["n_opp"] = n_opp
                qubit["phi_opp"] = phi_opp
                qubit["c_ops"] = c_ops
                qubit["t_g"] = t_g
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
    return gate_params


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
    H0, n_opp, phi_opp, c_ops, t_g, base_ex, base_size = qbi.init_qubit(qubit["Ec"], qubit["El"], qubit["Ej"], qubit["phi_dc"],qubit['omega_01'],qubit['alpha'], qubit["c_ops"], qubit["t_g"], truncation=qubit["truncation"], base_ex=qubit["base_ex"], base_size=qubit["base_size"])
    qubit["base_ex"] = base_ex
    qubit["base_size"] = base_size
    if H0 is None:
        print("H0 is None")
        return None
    #! is it diagonalized?
    qubit["H0"] = H0
    if len(H0) <= 2: qubit['alpha'] = np.inf
    else: qubit['alpha'] = H0[2,2]-H0[1,1] - (H0[1,1]-H0[0,0])
    qubit["omega_01"] = H0[1,1]-H0[0,0]
    qubit["n_opp"] = n_opp
    qubit["phi_opp"] = phi_opp
    qubit["c_ops"] = c_ops
    qubit["t_g"] = t_g
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
    gates = []
    scores = []
    cord = []
    #cnt = 0
    def loss_func(vals):
        global cnt
        local_gate = copy(gate_params)
        for i, key in enumerate(keys):
            exec(f"local_gate.{key}_amp = vals[i]")
        score, datapoint, local_gate = eval_qubit(local_gate, qubit, t0_samplerate=3, calZ=True)
        gates.append(local_gate)
        scores.append(-score)
        cord.append(vals)
        if 1 < 0:
            plt.clf()
            cmap = plt.get_cmap("viridis")
            plt.scatter(np.array(cord).T[0], np.array(cord).T[1], c=np.array(scores)+1, cmap=cmap)
            plt.colorbar()
            plt.xlabel(keys[0])
            plt.ylabel(keys[1])
            plt.title(f"Calibration for {qubit['name']} with {gate}")
            plt.savefig(f"temp/optimize_{qubit['name']}_{gate}.png")
            plt.savefig(f"temp/optimize.png")
            plt.clf()
        cnt += 1
        print(f"Count: {cnt}")
        if 1-score <= 2e-6:
            raise StopIteration
        return -score
    #do a simple scipy minimize
    try:
        _ = minimize(loss_func, initial_values, options={"maxiter": 20},method="Nelder-Mead")
    except StopIteration:
        print("Target fidelity reached")
    """final = result.x
    for i, key in enumerate(keys):
        exec(f"gate_params.{key}_amp = final[i]")"""
    gate_params = gates[np.argmin(scores)]
    if do_VZ:
        _, _, gate_params = eval_qubit(gate_params, qubit, t0_samplerate=3, calZ=True)
    
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
    return gate_params

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
    H0, n_opp, phi_opp, c_ops, t_g, base_ex, base_size = qbi.init_qubit(qubit["Ec"], qubit["El"], qubit["Ej"], qubit["phi_dc"],qubit['omega_01'],qubit['alpha'], qubit["c_ops"], qubit["t_g"],truncation=qubit["truncation"], base_ex=qubit["base_ex"], base_size=qubit["base_size"])
    qubit["base_ex"] = base_ex
    qubit["base_size"] = base_size
    if H0 is None:
        raise ValueError("H0 is None")
    #! is it diagonalized?
    scores, points = [], []
    qubit["H0"] = H0
    if len(H0) <= 2: qubit['alpha'] = np.inf
    else: qubit['alpha'] = H0[2,2]-H0[1,1] - (H0[1,1]-H0[0,0])
    qubit["omega_01"] = H0[1,1]-H0[0,0]
    qubit["n_opp"] = n_opp
    qubit["phi_opp"] = phi_opp
    qubit["c_ops"] = c_ops
    qubit["t_g"] = t_g
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
        gate_params = single_param(gate_params, qubit, gate, do_VZ)
    elif len(keys_wo_VZ) == 2:
        gate_params = two_param(gate_params, qubit, gate, do_VZ)
    elif len(keys_wo_VZ) == 0:
        val = gate_params.Omega_eq.subs(sp.Symbol("t_g"),qubit["t_g"])
        exec(f"gate_params.Omega_amp = val")
        #one for lambda, one for wether omega_{01} is in Omega_eq
        t1 = hasattr(gate_params,"lambda_eq")
        t2 = sp.Symbol("omega_{01}") in gate_params.Omega_eq.free_symbols
        if t1 or t2:
            H0, n_opp, phi_opp, c_ops, t_g, base_ex, base_size = qbi.init_qubit(qubit["Ec"], qubit["El"], qubit["Ej"], qubit["phi_dc"],qubit['omega_01'],qubit['alpha'], qubit["c_ops"], qubit["t_g"],truncation=qubit["truncation"], base_ex=qubit["base_ex"], base_size=qubit["base_size"])
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