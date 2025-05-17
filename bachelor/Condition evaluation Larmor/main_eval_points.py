import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

import sys
import numpy as np
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(os.getcwd())
from utils.gate_params import get_gate_params

config = {}
with open("config.txt", "r") as f:
    for line in f.readlines():
        line = line.strip()
        config[line.split(":")[0]] = eval(line.split(":")[1])

gate_names_2_eval = [#actual values are hidden off in the other file

    #"commensurate_x_virt_z_nooptim",
    
    
    
    #"corotating_xy_virt_z",
    #"FAST-MAGNUS_nooptim",
    "magnus1_x_virt_z_nooptim",
    "corotating_xy_virt_z_nooptim",
    "RWA_x_nooptim",
    "commensurate_x_virt_z",
    
    
    
    #"magnus1_x_virt_z",
    

    
    
    
]

scores_withmeta = {}
if os.path.exists("scores_withmeta.pickle"):
    with open("scores_withmeta.pickle", "rb") as f:
        scores_withmeta = pickle.load(f)


from utils.eval_functions import *
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from multiprocessing import Pool as normalPool
from copy import deepcopy as copy
from random import shuffle
from utils.eval_functions import AdaptiveLearner

#!todo: make sure that foound n_opp and phi_opp are normalized in their 0,1 entry

def instantiate_learners(gate_name):
    learner = AdaptiveLearner(scores_withmeta[gate_name], [10, 1, 10, None, None, np.pi, (10,10000), 0, 0], framework="adaptive_area",truncation=2)
    return learner

def main():
    #instantiate adaptive learner
    learners = {}
    """for k in list(scores_withmeta.keys()):
        if "commensurate" in k and not "nooptim" in k:
            scores_withmeta["comm_old"] = scores_withmeta[k]
            del scores_withmeta[k]
    #pickle
    with open("scores_withmeta.pickle", "wb") as f:
        pickle.dump(scores_withmeta, f)"""
    
    
    for i in range(len(gate_names_2_eval)):
        if gate_names_2_eval[i] not in scores_withmeta.keys():
            scores_withmeta[gate_names_2_eval[i]] = []
        #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [1.3, (0.1, 10), (0.1, 10), np.pi, (0.1,10), 1.58e-4, 1.58e-4], framework="adaptive")
        #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [1.3, (0.58, 0.6), (5.70, 5.72), np.pi, (5,20), 1.58e-4, 1.58e-4], framework="adaptive")
        #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [1.3, 0.59, 5.71, np.pi, (5,20), 1.58e-4, 1.58e-4], framework="adaptive")
        #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [1.3, 0.59, (0.1,10), np.pi, (5,20), 1.58e-4, 1.58e-4], framework="adaptive")
        #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [1.3, 0.59, (0.1,10), np.pi, (10,30), 1.58e-4, 1.58e-4], framework="adaptive")
        #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [1.3, (0,2), 5.71, np.pi, (5,20), 1.58e-4, 1.58e-4], framework="adaptive")
        #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [1.3, np.linspace(0,10,5), 5.71, np.pi, (10,30), 0, 0], framework="adaptive",truncation=2)
        
        #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [10, 1, 10, np.pi, (5,5.002), 0, 0], framework="adaptive",truncation=2)
        #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [10, 1, 10, np.pi, (1,1.002), 0, 0], framework="adaptive",truncation=2)
        #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [1, 0.8, 10, np.pi, (0.001,100), 0, 0], framework="adaptive",truncation=10)
        #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [1, 1, 1, np.pi, (29.4,29.6), 0, 0], framework="adaptive",truncation=2)
        #learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [None, 1, None, 0.25, 25, np.pi, (40,5000), 0, 0], framework="adaptive_area",truncation=3)
        learner = AdaptiveLearner(scores_withmeta[gate_names_2_eval[i]], [None, 1, None, 0.25, 25, np.pi, (40,5000), 0, 0], framework="adaptive_area",truncation=3)
        learners[gate_names_2_eval[i]] = learner
    #pool the above task
    
    """with normalPool(processes=1) as pool:#len(gate_names_2_eval)) as pool:
        results = pool.map(instantiate_learners, gate_names_2_eval)
        for i in range(len(gate_names_2_eval)):
            learners[gate_names_2_eval[i]] = results[i]"""
    while True:
        pointscnt = [len(scores_withmeta[gate_names_2_eval[i]]) if gate_names_2_eval[i] in scores_withmeta.keys() else 0 for i in range(len(gate_names_2_eval))]
        lowest = np.argsort(pointscnt)[:2]
        for k,gate_name in enumerate(gate_names_2_eval):
            if k not in lowest:
                continue
            #get the next point to evaluate
            learner = learners[gate_name]
            qubits_2_eval = learner.get_next_dp(N=64)
            shuffle(qubits_2_eval)
            gate_instances = []
            gate_params = get_gate_params(gate_name)
            for i in range(len(qubits_2_eval)):
                gate_instances.append(copy(gate_params))
            print(f"Evaluating {gate_name} with {len(qubits_2_eval)} qubits")
            if 1<0:
                for i,qubit,gate in zip(range(len(qubits_2_eval)), qubits_2_eval, gate_instances):
                    qubits_2_eval[i], gate_instances[i] = calib_gate((qubit, gate))
            else:
                results = []
                with normalPool(processes=16) as pool:
                    args = [(qubits_2_eval[i], gate_instances[i]) for i in range(len(qubits_2_eval))]
                    results = pool.map(calib_gate, args)
                    for i in range(len(qubits_2_eval)):
                        #gate_instances[i] = results[i]
                        qubits_2_eval[i], gate_instances[i] = results[i]
            print("Starting evaluation")

            if 1<0:
                results = [do_test(q,g) for q,g in zip(qubits_2_eval, gate_instances)]
            else:
                results = []
                with ProcessPool(max_workers=16) as pool:
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
                if gate_name not in scores_withmeta.keys():
                    scores_withmeta[gate_name] = []
                for j,point in enumerate(points):
                    #point["gate"] = gate_name
                    scores_withmeta[gate_name].append(point)
            with open(f"scores_withmeta.pickle", "wb") as f:
                pickle.dump(scores_withmeta, f)
            #feed the datapoints to the learner
            learner.feed_points(scores_withmeta[gate_name])
            


if __name__ == "__main__":
    main()