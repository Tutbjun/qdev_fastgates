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
    def __init__(self, known_dps, bounds, framework="adaptive"):
        #known_dps are expected to be a list of dictionaries
        ys = [dp["score"] for dp in known_dps]
        xs = [[dp["Ec"], dp["El"], dp["Ej"], dp["phi_dc"],dp["t_g"],dp["L1"],dp["L2"]] for dp in known_dps]
        point_keys = ["Ec", "El", "Ej", "phi_dc", "t_g", "L1", "L2"]
        self.ys = np.array(ys)
        self.xs = np.array(xs)
        self.point_keys = point_keys

        #bounds are expected to be a list of tuples. If an ellement is of float type rather than tuple, it is accepted as a fixed value
        bnds = [b if isinstance(b, tuple) else None for b in bounds]
        fixed = [b if not isinstance(b, tuple) else None for b in bounds]
        self.bounds = bnds
        self.fixed = fixed    

        self.conv_bounds(bnds, self.xs)

        self.framework = framework
        if framework == "botorch":
            
            #convert to torch tensors
            self.xs = torch.tensor(self.xs, dtype=torch.float32)
            self.ys = torch.tensor(self.ys, dtype=torch.float32)
            
            self.learner = SingleTaskGP(
                train_X=self.xs,
                train_Y=self.ys
            )
            self.mll = ExactMarginalLogLikelihood(self.learner.likelihood, self.learner)
            fit_gpytorch_mll(self.mll)
        elif framework == "adaptive":
            dummy_function = lambda x: None
            self.learner = adaptive.LearnerND(dummy_function, self.bounds_actual)
            if len(self.xs) > 0:
                for x,y in zip(self.xs,self.ys):
                    self.learner.tell(x,y)
        else:
            raise NotImplementedError(f"Framework {framework} is not implemented")
    def get_next_dp(self, N=1):
        if self.framework == "botorch":
            #get the next point to evaluate
            #use the acquisition function to get the next point
            acq = LogExpectedImprovement(self.learner, best_f=self.ys.max())
            bounds = torch.tensor(self.bounds_actual, dtype=torch.float32)
        elif self.framework == "adaptive":
            points = self.learner.ask(N)
            #put fixed values in the points
            tpoints = []
            for point in points[0]:
                tpoint = []
                j = 0
                for i,f in enumerate(self.fixed):
                    if f is not None:
                        tpoint.append(f)
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
            #save the qubit names
            with open("qubit_names.pickle", "wb") as f:
                pickle.dump(qubit_names, f)

            #return these
            return rpoints
        else:
            raise NotImplementedError(f"Framework {self.framework} is not implemented")
    def feed_points(self, points):
        if self.framework == "adaptive":
            #points are expected to be a list of dictionaries
            for point in points:
                #convert to numpy array
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
                #get the relevant
                #point = point[self.eval_mask]
                #feed the point to the learner
                try:
                    self.learner.tell(point, score)
                except ValueError as e:
                    if "Point already in triangulation" in str(e):
                        print(f"Point {point} already in triangulation")
                    else:
                        raise ValueError(f"ValueError: {e}")
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



def eval_qb(gate_params, qubit, t0_samplerate=5, calZ=False):
    
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
        loop = np.linspace(0, (H0[1,1]-H0[0,0]).real**(-1), t0_samplerate)
        if np.mean(np.abs(loop)) >t_g*10:
            loop = np.linspace(0, t_g, t0_samplerate)
        #loop = np.zeros(5)
    scores = []
    Z_amps = []
    results_list = []
    for t0 in loop:
        if gate_params.known_t0: 
            n = t0
            print(f"Testing at n = {n}")
            t0 = None
        else:
            print(f"Testing at t0 = {t0}")
            n = None
        gate_dynamics = copy(gate_params).assert_H0(copy(H0))
        #print(n_opp, phi_opp)
        gate_dynamics = copy(gate_dynamics.assert_opps(copy(n_opp), copy(phi_opp)))
        gate_dynamics = copy(gate_dynamics.compile_as_Qobj())
        H0_sim, H_sim = copy(gate_dynamics.transform_2_rotating_frame(t_g,omega_01=copy(H0[1,1]-H0[0,0]),t0=t0, n=n))
        #H = gate_dynamics.get_QuTiP_compile()
        gs.init_sim(H0_sim, H_sim, c_opps, n_opp, phi_opp,initial_state="even")
        if "H_log.txt" in os.listdir() and config["H_log"]: os.remove("H_log.txt")
        if n != None:
            t0 = gate_dynamics.t_0
            t0 = t0.subs(sp.symbols("n"), n)
            t0 = t0.subs(sp.symbols("t_g"), t_g)
            t0 = t0.subs(sp.symbols("omega_{01}"), H0[1,1]-H0[0,0])
            t0 = t0.evalf()
        results = gs.simulate(t0,1.1*t_g+t0)
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
            r = minimize(score_at_Zamp, 0.00, bounds=[(-1,1)])
            val_best = r.x[0]
            Z_amps.append(val_best)
        results_list.append(results)
    if calZ:
        Z_amp_avg = np.mean(Z_amps)
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
                lines = np.array([[float(line.strip().split(";")[0]),complex(line.strip().split(";")[1])] for line in lines], dtype=complex)
                X = lines.T[0]
                Y = lines.T[1]
                plt.scatter(X, Y.real, s=0.1)
                plt.scatter(X, Y.imag, s=0.1)
                plt.xlabel("Time")
                plt.ylabel("H(t)")
                plt.title(f"H(t) for {qubit['name']} with {gate}")
                plt.savefig(f"temp/H_log_{qubit['name']}_{gate}_t0_{t0}.png")
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
            gate_in_matrix = True
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
            #next actually do the calibration
            if not gate_in_matrix or not qb_in_matrix:
                #create a new matrix
                
                #do the calibration
                H0, n_opp, phi_opp, c_ops, t_g = qbi.init_qubit(qubit["Ec"], qubit["El"], qubit["Ej"], qubit["phi_dc"], qubit["c_ops"], qubit["t_g"])
                if H0 is None:
                    print("H0 is None")
                    return None
                #! is it diagonalized?
                qubit["H0"] = H0
                qubit['alpha'] = H0[2,2]-H0[1,1] - (H0[1,1]-H0[0,0])
                qubit["n_opp"] = n_opp
                qubit["phi_opp"] = phi_opp
                qubit["c_ops"] = c_ops
                qubit["t_g"] = t_g
                dummyfunc = lambda x: None
                #loss_per_interval = lambda xs,ys: np.diff(xs)*(np.mean(ys)**(-1))#!I have no clue what y is at this point, but this seem to give the correct behavior
                bnds = (0, 2)
                #loss_per_interval = lambda xs,ys: (np.power(np.diff(xs)/(bnds[1]-bnds[0]),-1))*(1.05-np.mean(ys))*np.max(np.array(xs)+1)*(np.min(np.append(-np.diff(ys).flatten(),0))+0.5)
                #loss_per_interval = lambda xs,ys: np.piecewise(
                #    (xs,ys),
                #    (ys[1]-ys[0] < 0 & np.isclose(xs[0],0),True),
                #    (-np.infty,lambda xs,ys: (np.power(np.diff(xs)/(bnds[1]-bnds[0]),-1))*(1.05-np.mean(ys))*np.mean(np.array(xs)+1))
                #)
                def loss_per_interval(xs,ys):
                    if np.isclose(xs[0],0) and ys[1]-ys[0] < 0:
                        return -np.inf
                    else:
                        return (np.power(np.diff(xs)/(bnds[1]-bnds[0]),-1))*(1.05-np.mean(ys))*np.mean(np.array(xs)+1)
                #Sampler = adaptive.Learner1D(dummyfunc, (0, 10), loss_per_interval=loss_per_interval)
                Sampler = Adaptive_1D_Custom(dummyfunc, bnds, loss_per_interval=loss_per_interval)
                for i in range(12):#!temp
                    point = Sampler.ask(1)
                    exec(f"gate_params.{valname}_amp = point[0][0]")
                    score, datapoint, gate_params = eval_qb(gate_params, qubit, t0_samplerate=3, calZ=do_VZ)
                    Sampler.tell(point[0][0], score)
                    x,y = [],[]
                    if (i%4 == 0 and i>10) or i == 9:
                        #find the peak, fit a polynomial, and eval the peak
                        data = Sampler.to_numpy()
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
                            score, datapoint, _ = eval_qb(gate_params, qubit, t0_samplerate=3, calZ=do_VZ)
                            Sampler.tell(peak_val, score)

                    data = Sampler.to_numpy()
                    points = data.T[0]
                    scores = data.T[1]
                    plt.plot(x, y)
                    plt.scatter(points, scores)
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
    #return gate_params

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
    H0, n_opp, phi_opp, c_ops, t_g = qbi.init_qubit(qubit["Ec"], qubit["El"], qubit["Ej"], qubit["phi_dc"], qubit["c_ops"], qubit["t_g"])
    if H0 is None:
        print("H0 is None")
        return None
    #! is it diagonalized?
    qubit["H0"] = H0
    qubit['alpha'] = H0[2,2]-H0[1,1] - (H0[1,1]-H0[0,0])
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
            initial = initial.subs(sp.Symbol('t_g'), t_g)
            initial = initial.subs(sp.Symbol('omega_{01}'), H0[1,1]-H0[0,0])
        initial_values[i] = initial
    gates = []
    def loss_func(vals):
        local_gate = copy(gate_params)
        for i, key in enumerate(keys):
            exec(f"local_gate.{key}_amp = vals[i]")
        score, datapoint, local_gate = eval_qb(local_gate, qubit, t0_samplerate=3, calZ=do_VZ)
        gates.append(local_gate)
        return -score
    #do a simple scipy minimize
    result = minimize(loss_func, initial_values, options={"maxiter": 20})

    
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
        pass
    #return gate_params