#use qutip to simulate the gate on every coordinate dirreciton given a set of pulse parameters


import qutip as qt
import numpy as np
import matplotlib.pyplot as plt



start_states = []
solvers = []
t0s = []

def init_sim(H0, H,c_ops, n_opp,phi_opp, initial_state="even"):
    #qbi.init_qubit(1.3,0.59,5.71,np.pi)
    global start_states
    global solvers
    global t0s
    stateCnt = len(H0)
    paulis = [qt.sigmax(),qt.sigmay(),qt.sigmaz()]
    start_states = []
    if initial_state == "even":
        #for each pauli-matrix eigenstate
        for i in range(3):
            eigen = np.linalg.eig(paulis[i].full())
            for e in eigen[1].T:
                #expand the basis and append
                state = np.zeros(stateCnt,dtype=np.complex128)
                state[:len(e)] = e
                start_states.append(state)
    else:
        raise NotImplementedError
    omega_01 = H0[1,1]-H0[0,0]
    #print(H(0).full())
    #print(H(5).full())
    #make plot of |0><1| on H
    """T = np.linspace(0, 20, 100)
    r = [H(t).full()[0][1].real for t in T]
    i = [H(t).full()[0][1].imag for t in T]
    plt.plot(T, r)
    plt.plot(T, i)
    plt.savefig("temp/envelope.png")
    plt.clf()
    plt.close('all')"""

    tau = 1/omega_01
    tau_r = np.real(tau)
    #print(f"tau-tau_r error: {tau-tau_r}")
    tau = tau_r
    """if initial_time == "random":
        t0s = [np.random.rand()*tau]
    elif initial_time == "even":
        t0s = np.linspace(0,tau,5)#arbitrary sample-rate=5
    else:
        raise NotImplementedError"""
    #pure to density
    for i in range(len(start_states)):
        start_states[i] = qt.ket2dm(qt.Qobj(start_states[i][:,np.newaxis]))
    solvers = []
    for i in range(len(start_states)):
            
        solvers.append(qt.MESolver(H,c_ops=c_ops, options={'store_final_state':True, 'progress_bar': "tqdm", "method": "bdf", "max_step": 2}))#, "min_step": tau/100}))
        #print(H(0).full()-H(0).full().T.conj())
        #solvers.append(qt.MESolver(H, c_ops=[], options={'nsteps':10
                                    #args for run:
                                    #, start_states[i], np.linspace(t0,tau*2+t0,1000), [], []))
        #solvers.append(qt.SESolver(H, options={'nsteps':10000}))
    

import asyncio
import multiprocessing
def simulate(t0,t_length):
    global solvers

    results = []
    i = 0
    for j in range(len(start_states)):
        #print(start_states[j].full())
        result = solvers[i].run(start_states[j], np.linspace(t0,t_length,500))
        """process = multiprocessing.Process(target=solvers[i].run, args=(start_states[j], np.linspace(t0,t_length+t0,1000), [], []))
        process.start()
        process.join(timeout=timeout)
        if process.is_alive():
            print(f"Process {i} timed out")
            process.terminate()
            process.join()
            result = None
        else:
            result = solvers[i].results
        i += 1"""
        results.append(result)
        #print(result.states[0].full())
    return results


#init_sim()