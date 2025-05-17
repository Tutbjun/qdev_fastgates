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
from utils.gate_params import get_gate_params

config = {}
with open("config.txt", "r") as f:
    for line in f.readlines():
        line = line.strip()
        config[line.split(":")[0]] = eval(line.split(":")[1])

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


from ..utils.eval_functions import *
from pebble import ProcessPool
from concurrent.futures import TimeoutError
from multiprocessing import Pool as normalPool
from copy import deepcopy as copy
from random import shuffle
from bachelor.utils.eval_functions import AdaptiveLearner
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
        learner = AdaptiveLearner(scheme_scores_4_adaptive[gate_names_2_eval[i]], [1.3, 0.59, (0.1,10),None,None, np.pi, (10,30), 1.58e-4, 1.58e-4], framework="adaptive")
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
                    gate_instances[i] = calib_gate((qubit, gate))
            else:
                results = []
                with normalPool(processes=15) as pool:
                    args = [(qubits_2_eval[i], gate_instances[i]) for i in range(len(qubits_2_eval))]
                    results = pool.map(calib_gate, args)
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
                if gate_name not in scheme_scores_4_adaptive.keys():
                    scheme_scores_4_adaptive[gate_name] = []
                for j,point in enumerate(points):
                    #point["gate"] = gate_name
                    scheme_scores_4_adaptive[gate_name].append(point)
            with open(f"scheme_scores_4_adaptive.pickle", "wb") as f:
                pickle.dump(scheme_scores_4_adaptive, f)
            #feed the datapoints to the learner
            learner.feed_points(scheme_scores_4_adaptive[gate_name])
            


if __name__ == "__main__":
    main()