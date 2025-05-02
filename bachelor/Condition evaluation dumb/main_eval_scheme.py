import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
#TODO:
# #!1. Maybe use adaptive sampling instead of binhopping to solve the hangup
# #!2. Make sure gate parameters are set in a practical manner
# #!3. Test out the circularly polarized gate
#!4. Test out the commensurate gate
# #!5. Figure an automation out such that one does not have to retest gates (one landscape per gate?)
# #!6. Put this whole thing on git


import sys
import numpy as np
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gate_params import get_gate_params

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
    #"demo_x"
    #"corotating_xy",
    #"corotating_xy_virt_z",
    #"simple_x",
    #"commensurate_x_virt_z",
    "magnus1_x_virt_z"
]
scheme_scores = {}
scheme_scores_raw = []
scheme_scores_4_adaptive = {}
if os.path.exists("scheme_scores_raw.pickle"):
    with open("scheme_scores_raw.pickle", "rb") as f:
        scheme_scores_raw = pickle.load(f)
if os.path.exists("scheme_scores_4_adaptive.pickle"):
    with open("scheme_scores_4_adaptive.pickle", "rb") as f:
        scheme_scores_4_adaptive = pickle.load(f)




"""from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
import torch"""

from multiprocessing import Pool

def do_test(qubit,gate_params):
    H0, n_opp, phi_opp, c_ops, t_g = qbi.init_qubit(qubit["Ec"], qubit["El"], qubit["Ej"], qubit["phi_dc"], qubit["c_ops"], qubit["t_g"])
    if H0 is None:
        raise ValueError("H0 is None")
    #! is it diagonalized?
    scores, points = [], []
    qubit["H0"] = H0
    qubit['alpha'] = H0[2,2]-H0[1,1] - (H0[1,1]-H0[0,0])
    qubit["n_opp"] = n_opp
    qubit["phi_opp"] = phi_opp
    qubit["c_ops"] = c_ops
    qubit["t_g"] = t_g
    #for gate in gate_names_2_eval:
    #    #gate_params = get_gate_params(gate)
    score, datapoint, _ = eval_qb(gate_params, qubit)
    scores.append(score)
    points.append(datapoint)
    return scores, points

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
                losses[i] = self.loss_per_interval((self.xs[i], self.xs[i+1]), (self.ys[i], self.ys[i+1]))
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


from scipy.optimize import basinhopping
import eval_functions as cf
def do_calib(args):
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
    if len(keys_wo_VZ) <= 1:#singel variable calibration
        gate_params = cf.single_param(gate_params, qubit, gate, do_VZ)
    elif len(keys_wo_VZ) == 2:
        gate_params = cf.two_param(gate_params, qubit, gate, do_VZ)
    return gate_params



from pebble import ProcessPool
from concurrent.futures import TimeoutError
from multiprocessing import Pool as normalPool
from copy import deepcopy as copy
from random import shuffle
from eval_functions import AdaptiveLearner
def main():
    #instantiate adaptive learner
    learners = {}
    for i in range(len(gate_names_2_eval)):
        if gate_names_2_eval[i] not in scheme_scores_4_adaptive.keys():
            scheme_scores_4_adaptive[gate_names_2_eval[i]] = []
        #learner = AdaptiveLearner(scheme_scores_4_adaptive[gate_names_2_eval[i]], [1.3, (0.1, 10), (0.1, 10), np.pi, (0.1,10), 1.58e-4, 1.58e-4], framework="adaptive")
        #learner = AdaptiveLearner(scheme_scores_4_adaptive[gate_names_2_eval[i]], [1.3, (0.58, 0.6), (5.70, 5.72), np.pi, (5,20), 1.58e-4, 1.58e-4], framework="adaptive")
        #learner = AdaptiveLearner(scheme_scores_4_adaptive[gate_names_2_eval[i]], [1.3, 0.59, 5.71, np.pi, (5,20), 1.58e-4, 1.58e-4], framework="adaptive")
        #learner = AdaptiveLearner(scheme_scores_4_adaptive[gate_names_2_eval[i]], [1.3, 0.59, (0.1,10), np.pi, (5,20), 1.58e-4, 1.58e-4], framework="adaptive")
        learner = AdaptiveLearner(scheme_scores_4_adaptive[gate_names_2_eval[i]], [1.3, 0.59, (0.1,10), np.pi, (10,30), 1.58e-4, 1.58e-4], framework="adaptive")
        #learner = AdaptiveLearner(scheme_scores_4_adaptive[gate_names_2_eval[i]], [1.3, (0,2), 5.71, np.pi, (5,20), 1.58e-4, 1.58e-4], framework="adaptive")
        learners[gate_names_2_eval[i]] = learner
    while True:
        for gate_name in gate_names_2_eval:
            #get the next point to evaluate
            learner = learners[gate_name]
            qubits_2_eval = learner.get_next_dp(N=30)
            shuffle(qubits_2_eval)
            gate_instances = []
            gate_params = get_gate_params(gate_name)
            for i in range(len(qubits_2_eval)):
                gate_instances.append(copy(gate_params))
            print(f"Evaluating {gate_name} with {len(qubits_2_eval)} qubits")
            if 1<0:
                for i,qubit,gate in zip(range(len(qubits_2_eval)), qubits_2_eval, gate_instances):
                    gate_instances[i] = do_calib((qubit, gate))
            else:
                results = []
                with normalPool(processes=15) as pool:
                    args = [(qubits_2_eval[i], gate_instances[i]) for i in range(len(qubits_2_eval))]
                    results = pool.map(do_calib, args)
                    for i in range(len(qubits_2_eval)):
                        gate_instances[i] = results[i]
            print("Starting evaluation")

            if 1<0:
                results = [do_test(q,g) for q,g in zip(qubits_2_eval, gate_instances)]
            else:
                results = []
                with ProcessPool(max_workers=15) as pool:
                    
                    future = pool.map(do_test, qubits_2_eval, gate_instances, timeout=60*60)#!conservative
                    iter = future.result()
                    for i in range(len(qubits_2_eval)): 
                        def get_null_result(q,timeout=False):
                            result = [[None], None]
                            result[1] = [q]
                            result[1][0]["score"] = None
                            if timeout:
                                result[1][0]["timeout"] = True
                            return result
                        try:
                            result = next(iter)
                            results.append(result)
                        except ValueError as e:
                            print(f"ValueError: {e}")
                            results.append(get_null_result(qubits_2_eval[i]))
                        except TimeoutError:
                            print("Timeout")
                            results.append(get_null_result(qubits_2_eval[i], timeout=True))
                        except:
                            print("Unknown error")
            #unpack the results
            for i,result in enumerate(results):
                scores, points = result
                #index_best = 0
                #gate_best = ""
                #for j,point in enumerate(points):
                #    scheme_scores_raw.append(point)
                #    if point["score"] is not None:
                #        if point["score"] >= points[index_best]["score"]:
                #            index_best = j
                #            gate_best = gate_names_2_eval[j]
                #best_point = points[index_best]
                #best_point["gate_winner"] = gate_best
                #scheme_scores_4_adaptive.append(best_point)
                if gate_name not in scheme_scores_4_adaptive.keys():
                    scheme_scores_4_adaptive[gate_name] = []
                for j,point in enumerate(points):
                    #point["gate"] = gate_name
                    scheme_scores_4_adaptive[gate_name].append(point)
                    
                
                """for j,s in enumerate(scores):
                    q_name = qubits_2_eval[i]["name"]
                    g_name = gate_names_2_eval[j]
                    if q_name not in scheme_scores.keys(): scheme_scores[q_name] = {}
                    scheme_scores[q_name][g_name] = s"""
            #print(scheme_scores)
            #! save the results

            """for k1 in scheme_scores.keys():
                for k2  in scheme_scores[k1].keys():
                    val = scheme_scores[k1][k2]
                    if val != None:
                        scheme_scores[k1][k2] = val.real
                    else:
                        scheme_scores[k1][k2] = None
            with open("scheme_scores.json", "w") as f:
                json.dump(scheme_scores, f)"""
            
            """with open("scheme_scores_raw.pickle", "wb") as f:
                pickle.dump(scheme_scores_raw, f)"""
            with open(f"scheme_scores_4_adaptive.pickle", "wb") as f:
                pickle.dump(scheme_scores_4_adaptive, f)
            #feed the datapoints to the learner
            learner.feed_points(scheme_scores_4_adaptive[gate_name])
            


if __name__ == "__main__":
    main()