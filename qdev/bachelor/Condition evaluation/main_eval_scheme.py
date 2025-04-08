import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
#TODO:
#!1. Maybe use adaptive sampling instead of binhopping to solve the hangup
#!2. Make sure gate parameters are set in a practical manner
#!3. Test out the circularly polarized gate
#!4. Test out the commensurate gate
#!5. Figure an automation out such that one does not have to retest gates (one landscape per gate?)
#!6. Put this whole thing on git


import sys
import numpy as np
import json
import matplotlib.pyplot as plt
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gate_params import get_gate_params
from utils import gate_evaluator as ge
from utils import gate_simulator as gs
from utils import param_object as po
from utils import qubit_init as qbi
from utils import hardware_attenuation as ha

config = {}
with open("config.txt", "r") as f:
    for line in f.readlines():
        line = line.strip()
        config[line.split(":")[0]] = eval(line.split(":")[1])

"""qubits_2_eval = {
    "qubit_1": {
        "Ec": 1.3,
        "El": 0.59,
        "Ej": 5.71,
        "t_g": 1,
        "phi_dc": np.pi,
        "name": "qubit_1",
        "c_ops": [
            np.array([[0,1.58e-4],[0,0]]),#
            np.array([[1.5811388300841896659994467722164e-4,0],[0,-1.5811388300841896659994467722164e-4]])
        ]
    }
}"""
gate_names_2_eval = [#actual values are hidden off in the other file
    "demo_x"
]
scheme_scores = {}
scheme_scores_raw = []
scheme_scores_4_adaptive = []
if os.path.exists("scheme_scores_raw.pickle"):
    with open("scheme_scores_raw.pickle", "rb") as f:
        scheme_scores_raw = pickle.load(f)
if os.path.exists("scheme_scores_4_adaptive.pickle"):
    with open("scheme_scores_4_adaptive.pickle", "rb") as f:
        scheme_scores_4_adaptive = pickle.load(f)

import asyncio
from copy import deepcopy as copy
def eval_qb(gate_params, qubit, t0_samplerate=5):
    
    #global scheme_scores
    #global scheme_scores_raw
    global config
    gate = gate_params.name
    #set params (omega_01, omega_12, etc.)
    #import envelope
    H0, n_opp, phi_opp, t_g = qubit["H0"], qubit["n_opp"], qubit["phi_opp"], qubit["t_g"]
    c_opps = qubit["c_ops"]
    
    #t_g = gate_params.t_g
    ideal_gate = gate_params.ideal
    #ideal_gate = get_gate_params("ideal")
    loop = []
    if gate_params.known_t0:
        loop = [gate_params.t_0]
    else:
        loop = np.linspace(0, (H0[1,1]-H0[0,0]).real**(-1), t0_samplerate)
        #loop = np.zeros(5)
    scores = []
    for t0 in loop:#!temp until bug is resolved
        print(f"Testing at t0 = {t0}")
        gate_dynamics = copy(gate_params).assert_H0(copy(H0))
        gate_dynamics = copy(gate_dynamics.assert_opps(copy(n_opp), copy(phi_opp)))
        gate_dynamics = copy(gate_dynamics.compile_as_Qobj())
        H0_sim, H_sim = copy(gate_dynamics.transform_2_rotating_frame(t0,omega_01=copy(H0[1,1]-H0[0,0])))
        #H = gate_dynamics.get_QuTiP_compile()
        gs.init_sim(H0_sim, H_sim, c_opps, n_opp, phi_opp,initial_state="even")
        if "H_log.txt" in os.listdir() and config["H_log"]: os.remove("H_log.txt")

        results = gs.simulate(t0,1.1*t_g+t0)
        for i in range(len(results)):
            results[i].solver = f"{qubit['name']}_{gate}_t0_{t0}_i_{i}"
        score = ge.evaluate(results,ideal_gate)
        scores.append(score)
        if config["H_log"]:
            with open("H_log.txt", "r") as f:
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
        "score": s,
        "gate": gate,
        "L1": qubit["L1"],
        "L2": qubit["L2"],
    }
    #scheme_scores_raw.append(dp)
    return s,dp







"""from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
import torch"""
import adaptive
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
            if k == "L2":
                L2 = point[k]
                c2 = np.array([[L2,0],[0,L2]])
        point["c_ops"] = [c1, c2]
        return point

from multiprocessing import Pool

def do_test(qubit,gate_params):
    H0, n_opp, phi_opp, c_ops, t_g = qbi.init_qubit(qubit["Ec"], qubit["El"], qubit["Ej"], qubit["phi_dc"], qubit["c_ops"], qubit["t_g"])
    if H0 is None:
        raise ValueError("H0 is None")
    #! is it diagonalized?
    scores, points = [], []
    qubit["H0"] = H0
    qubit["n_opp"] = n_opp
    qubit["phi_opp"] = phi_opp
    qubit["c_ops"] = c_ops
    qubit["t_g"] = t_g
    for gate in gate_names_2_eval:
        gate_params = get_gate_params(gate)
        score, datapoint = eval_qb(gate_params, qubit)
    scores.append(score)
    points.append(datapoint)
    return scores, points

from scipy.optimize import basinhopping
def do_calib(qubit, gate):
    #do the calibration
    #get the gate params
    gate_params = get_gate_params(gate)
    #check keys to calibrate
    for key in gate_params.is_calibrated.keys():
        if gate_params.is_calibrated[key] == False:
            gate_in_matrix = True
            qb_in_matrix = True
            #first check if (A) calibration matrix exists, and (B) if it has an entry for this gate/qubit
            gate_name = gate_params.name
            matrixname = f"temp/calibration_matrix_val={key}.pickle"
            if os.path.exists(matrixname):
                with open(matrixname, "rb") as f:
                    matrix = pickle.load(f)
                if gate_name in matrix.keys():
                    #check if the qubit is in the matrix
                    if qubit["index"] in matrix[gate_name].keys():
                        #get the value
                        val = matrix[gate_name][qubit["index"]]
                        #assert the value
                        #gate_params.assert_calibration(key, val)
                        exec(f"gate_params.{valname}_amp = val")
                    else:
                        qb_in_matrix = False
                else:
                    gate_in_matrix = False
                    qb_in_matrix = False
            else:
                matrix = {}
                gate_in_matrix = False
                qb_in_matrix = False
            if not gate_in_matrix:
                #create a new matrix
                matrix[gate_name] = {}
                if not qb_in_matrix:
                    #create a new entry for the qubit
                    matrix[gate_name][qubit["index"]] = 0
            #next actually do the calibration
            if not gate_in_matrix or not qb_in_matrix:
                #create a new matrix
                valname = key
                val = eval(f"gate_params.{valname}_amp")
                gate_params
                qubit["index"]
                #do the calibration
                H0, n_opp, phi_opp, c_ops, t_g = qbi.init_qubit(qubit["Ec"], qubit["El"], qubit["Ej"], qubit["phi_dc"], qubit["c_ops"], qubit["t_g"])
                if H0 is None:
                    raise ValueError("H0 is None")
                #! is it diagonalized?
                qubit["H0"] = H0
                qubit["n_opp"] = n_opp
                qubit["phi_opp"] = phi_opp
                qubit["c_ops"] = c_ops
                qubit["t_g"] = t_g
                def optimization_task(val,qubit,gate_params,valname):
                    exec(f"gate_params.{valname}_amp = val")
                    print(f"Testing with {valname} = {val}")
                    score, datapoint = eval_qb(gate_params, qubit,t0_samplerate=3)
                    return 1-score
                result = basinhopping(optimization_task, val, niter=1, T=1e-1, stepsize=0.5, minimizer_kwargs={"args": (qubit, gate_params,valname)}, disp=True)
                #get the optimized value
                #vals = result.x
                #scores = result.fun
                #indx_best = np.argmin(scores)
                #val = vals[indx_best]
                val = result.x
                matrix[gate_name][qubit["index"]] = val
                #assert the value
                #gate_params.assert_calibration(key, val)
                exec(f"gate_params.{valname}_amp = val")
            #save the matrix
            with open(matrixname, "wb") as f:
                pickle.dump(matrix, f)
    #return gate_params
    return gate_params



from pebble import ProcessPool
from concurrent.futures import TimeoutError
def main():
    #instantiate adaptive learner
    learner = AdaptiveLearner(scheme_scores_4_adaptive, [1.3, (0.1, 10), (0, 10), np.pi, (0.1,10), 1.58e-4, 1.58e-4], framework="adaptive")
    while True:
        #get the next point to evaluate
        qubits_2_eval = learner.get_next_dp(N=64)
        #do potential calibration of gates
        gates_2_eval = np.zeros((len(qubits_2_eval), len(gate_names_2_eval))).tolist()
        i_s = range(len(qubits_2_eval))
        j_s = range(len(gate_names_2_eval))
        grid = np.meshgrid(i_s, j_s)
        calibs = [(i,qubits_2_eval[i],j,gate_names_2_eval[j]) for i,j in zip(grid[0].flatten(), grid[1].flatten())]
        """for i in range(len(qubits_2_eval)):
            gates_2_eval.append([])
            for j in range(len(gate_names_2_eval)):
                gates_2_eval[i].append(do_calib(qubits_2_eval[i], gate_names_2_eval[j]))
        raise NotImplementedError("Calibration not asserted")"""
        if 1<0:
            for i,qubit,j,gate in calibs:
                gates_2_eval[i][j] = do_calib(qubit, gate)
        else:
            results = []
            with ProcessPool(max_workers=16) as pool:
                future = pool.map(do_calib, [calibs[i][1] for i in range(len(calibs))], [calibs[i][3] for i in range(len(calibs))])
                iter = future.result()
                pool.close()
                pool.join()
                for i,j in zip(grid[0].flatten(), grid[1].flatten()):
                    try:
                        result = next(iter)
                        gates_2_eval[i][j] = result
                    except ValueError as e:
                        print(f"ValueError: {e}")
                    except TimeoutError:
                        print("Timeout")
                    except:
                        print("Unknown error")


        new_scheme_scores_raw = []
        if 1<0:
            results = [do_test(q,g) for q,g in zip(qubits_2_eval, gates_2_eval)]
        else:
            results = []
            with ProcessPool(max_workers=16) as pool:
                
                future = pool.map(do_test, qubits_2_eval, gates_2_eval, timeout=5*60)
                iter = future.result()
                for i in range(len(qubits_2_eval)): 
                    def get_null_result(q):
                        result = [[None], None]
                        result[1] = [q]
                        result[1][0]["score"] = None
                        return result
                    try:
                        result = next(iter)
                        results.append(result)
                    except ValueError as e:
                        print(f"ValueError: {e}")
                        results.append(get_null_result(qubits_2_eval[i]))
                    except TimeoutError:
                        print("Timeout")
                        results.append(get_null_result(qubits_2_eval[i]))
                    except:
                        print("Unknown error")
        #unpack the results
        for i,result in enumerate(results):
            scores, points = result
            index_best = 0
            gate_best = ""
            for j,point in enumerate(points):
                scheme_scores_raw.append(point)
                if point["score"] is not None:
                    if point["score"] >= points[index_best]["score"]:
                        index_best = j
                        gate_best = gate_names_2_eval[j]
            best_point = points[index_best]
            best_point["gate_winner"] = gate_best
            scheme_scores_4_adaptive.append(best_point)
            
            for j,s in enumerate(scores):
                q_name = qubits_2_eval[i]["name"]
                g_name = gate_names_2_eval[j]
                if q_name not in scheme_scores.keys(): scheme_scores[q_name] = {}
                scheme_scores[q_name][g_name] = s
        print(scheme_scores)
        #! save the results

        for k1 in scheme_scores.keys():
            for k2  in scheme_scores[k1].keys():
                val = scheme_scores[k1][k2]
                if val != None:
                    scheme_scores[k1][k2] = val.real
                else:
                    scheme_scores[k1][k2] = None
        with open("scheme_scores.json", "w") as f:
            json.dump(scheme_scores, f)
        
        with open("scheme_scores_raw.pickle", "wb") as f:
            pickle.dump(scheme_scores_raw, f)
        with open("scheme_scores_4_adaptive.pickle", "wb") as f:
            pickle.dump(scheme_scores_4_adaptive, f)
        #feed the datapoints to the learner
        learner.feed_points(scheme_scores_4_adaptive)
        


if __name__ == "__main__":
    main()