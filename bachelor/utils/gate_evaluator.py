import numpy as np
import qutip as qt
import matplotlib.pyplot as plt


def evaluate(results,ideal_gate):#!validate
    """
    Do a qutip simulation, where the inverse of the ideal gate is used to run the simulation back. Then take the amplitude |initial> in the final state
    """
    

    #print(results)
    propabilities = []
    #expand the basis
    base_len = len(results[0].states[0].full())
    ideal_gate_inverse = np.eye(base_len,base_len,dtype=complex)
    ideal_gate_inverse[:2,:2] = np.array(ideal_gate).conj().T

    sx_,sy_,sz_ = qt.sigmax(),qt.sigmay(),qt.sigmaz()
    #sx, sy, sz = np.eye(base_len,dtype=complex), np.eye(base_len,dtype=complex), np.eye(base_len,dtype=complex)
    sx, sy, sz = np.zeros((base_len,base_len),dtype=complex), np.zeros((base_len,base_len),dtype=complex), np.zeros((base_len,base_len),dtype=complex)
    sx[:2,:2], sy[:2,:2], sz[:2,:2] = sx_.full(), sy_.full(), sz_.full()



    for result in results:
        #plot states on bloch sphere
        ax=[0,0]
        fig = plt.figure()
        ax[0] = fig.add_subplot(121, projection='3d')
        ax[1] = fig.add_subplot(122)
        bloch = qt.Bloch(fig=fig, axes=ax[0])
        exp_x = [np.trace(np.dot(sx,state.full())) for state in result.states]
        exp_y = [np.trace(np.dot(sy,state.full())) for state in result.states]
        exp_z = [np.trace(np.dot(sz,state.full())) for state in result.states]
        bloch.add_points([exp_x,exp_y,exp_z],meth="l")
        bloch.show()
        ax[1].plot(range(len(result.states)), exp_z, label="X")
        s2,s3,s4 = np.zeros_like(sz), np.zeros_like(sz), np.zeros_like(sz)
        s2[2,2],s3[3,3],s4[4,4] = 1,1,1
        ax[1].plot(range(len(result.states)), [np.trace(np.dot(s2,state.full())) for state in result.states], label="|2>")
        ax[1].plot(range(len(result.states)), [np.trace(np.dot(s3,state.full())) for state in result.states], label="|3>")
        ax[1].plot(range(len(result.states)), [np.trace(np.dot(s4,state.full())) for state in result.states], label="|4>")
        try:
            plt.savefig(f"temp/bloch_{result.solver}.png")
            plt.savefig("bloch.png")
        except:
            print("Could not save figure")
        #plt.show()
        plt.clf()
        fig.clf()
        #fig.close()



        #get final state
        final_state = result.states[-1]
        #print("Final state: ",final_state.full())
        #get initial state
        initial_state = result.states[0]
        #the states are expressed in density matricies, so the opperators should be promoted to super opperators
        ideal_gate_inverse_L = qt.spre(qt.Qobj(ideal_gate_inverse))
        ideal_gate_inverse_R = qt.spost(qt.Qobj(ideal_gate_inverse))
        S = ideal_gate_inverse_L*ideal_gate_inverse_R#!validate this
        #apply the inverse gate
        #print("Trace: ",final_state.tr())
        #print("Trace-preserving check:", ideal_gate_inverse_L.tr(), ideal_gate_inverse_R.tr())
        #final_state = qt.vector_to_operator(S*qt.operator_to_vector(final_state))
        final_state = np.dot(np.dot(ideal_gate_inverse,final_state.full()),ideal_gate_inverse.T.conj())
        final_state = qt.Qobj(final_state)
        #find which opperator the initial state is an eigenvector of
        truth = initial_state.isherm and np.isclose(initial_state.tr(),1) and np.isclose((initial_state*initial_state).tr(),1)
        if not truth:
            print("The initial state is not a pure state!")
        #print("Trace: ",final_state.tr())
        #print(final_state.full())
        eigvals,eigkets = initial_state.eigenstates()
        eigkets,eigvals = [eigket.full() for i,eigket in enumerate(eigkets) if np.isclose(eigvals[i],1)], [eigval for eigval in eigvals if np.isclose(eigval,1)] 
        initial_pure,eigval = eigkets[0],eigvals[0]#!incompatible eigvals
        #print(initial_pure)
        initial_type = None
        if np.isclose(np.dot(sx,initial_pure),initial_pure).all():
            initial_type = "x"
            #exp = np.trace(np.dot(sx,final_state.full()))
            
        elif np.isclose(np.dot(sy,initial_pure),initial_pure).all():
            initial_type = "y"
            exp = np.trace(np.dot(sy,final_state.full()))
        elif np.isclose(np.dot(sz,initial_pure),initial_pure).all():
            initial_type = "z"
            exp = np.trace(np.dot(sz,final_state.full()))
        elif np.isclose(np.dot(sx,initial_pure),-initial_pure).all():
            initial_type = "-x"
            exp = np.trace(np.dot(sx,final_state.full()))
        elif np.isclose(np.dot(sy,initial_pure),-initial_pure).all():
            initial_type = "-y"
            exp = np.trace(np.dot(sy,final_state.full()))
        elif np.isclose(np.dot(sz,initial_pure),-initial_pure).all():
            initial_type = "-z"
            exp = np.trace(np.dot(sz,final_state.full()))
        else:
            print("The initial state is not an eigenvector of any of the Pauli opperators!")
        """if np.isclose(np.linalg.det(sx[:2,:2]-eigval*np.eye(2)),0).all():
            #print("x")
            prob = np.trace(np.dot(sx,final_state.full()))#!validate
        elif np.isclose(np.linalg.det(sy[:2,:2]-eigval*np.eye(2)),0).all():
            #print("y")
            prob = np.trace(np.dot(sy,final_state.full()))
        elif np.isclose(np.linalg.det(sz[:2,:2]-eigval*np.eye(2)),0).all():
            #print("z")
            prob = np.trace(np.dot(sz,final_state.full()))
        else:
            print("The initial state is not an eigenvector of any of the Pauli opperators!")
            prob = 0"""
        """if "x" in initial_type:
            prob = np.trace(np.dot(sx,final_state.full()))
        elif "y" in initial_type:
            prob = np.trace(np.dot(sy,final_state.full()))
        elif "z" in initial_type:
            prob = np.trace(np.dot(sz,final_state.full()))
        else:
            print("The initial state is not an eigenvector of any of the Pauli opperators!")"""
        fidelity = np.sum([initial_pure[i].conj()*initial_pure[i]*final_state.full()[i,i] for i in range(len(initial_pure))])
        print("Fidelity: ",fidelity)
        prob = np.trace(np.dot(initial_state.full().T.conj(),final_state.full()))#!why are expectation values complex?
        #!why are expectation values complex?
        propabilities.append(fidelity)
    fidelity = np.sum(propabilities)/len(propabilities)
    return fidelity