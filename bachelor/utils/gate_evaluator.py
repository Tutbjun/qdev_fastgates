import numpy as np
import qutip as qt
#import qutip_cupy
import matplotlib.pyplot as plt
import matplotlib._pylab_helpers as pylab_helpers
def is_figure_active():
    return len(pylab_helpers.Gcf.figs) > 0
config = {}
with open("config.txt", "r") as f:
    for line in f.readlines():
        line = line.strip()
        config[line.split(":")[0]] = eval(line.split(":")[1])

def evaluate(results,ideal_gate,light=False):#!validate
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
        initial_state = result.states[0]
        final_state = result.states[-1]
        final_state = np.dot(np.dot(ideal_gate_inverse,final_state.full()),ideal_gate_inverse.T.conj())
        final_state = qt.Qobj(final_state)
        nextlast_state = result.states[-2]
        #print((final_state-result.states[-2]).full())
        if config["bloch"] and not light:

            
            ax=[0,0]
            if not is_figure_active():
                fig = plt.figure()
                ax[0] = fig.add_subplot(121, projection='3d')
                ax[1] = fig.add_subplot(122)
                bloch = qt.Bloch(fig=fig, axes=ax[0])
                exp_x = [np.trace(np.dot(sx,state.full())) for state in result.states]
                exp_y = [np.trace(np.dot(sy,state.full())) for state in result.states]
                exp_z = [np.trace(np.dot(sz,state.full())) for state in result.states]
                bloch.add_points([exp_x,exp_y,exp_z],meth="l")
                ax[1].plot(range(len(result.states)), exp_x, label=r"$\sigma_x$")
                ax[1].plot(range(len(result.states)), exp_y, label=r"$\sigma_y$")
                ax[1].plot(range(len(result.states)), exp_z, label=r"$\sigma_z$")
                #plot final state as a red x on the bloch sphere
                fx = np.trace(np.dot(sx,final_state.full()))
                fy = np.trace(np.dot(sy,final_state.full()))
                fz = np.trace(np.dot(sz,final_state.full()))
                bloch.add_points([fx,fy,fz],meth="s",colors="red")
                bloch.add_points([exp_x[-1],exp_y[-1],exp_z[-1]],meth="s",colors="orange")
                nlx = np.trace(np.dot(sx,nextlast_state.full()))
                nly = np.trace(np.dot(sy,nextlast_state.full()))
                nlz = np.trace(np.dot(sz,nextlast_state.full()))
                bloch.add_points([nlx,nly,nlz],meth="s",colors="green")
                bloch.show()
                ax[1].set_xlabel("Timesteps")
                ax[1].set_ylabel("Expectation value")
                ax[1].legend()
            
                try:
                    #plt.savefig(f"temp/bloch_{result.solver}.png")
                    plt.savefig("bloch.png")
                    pass
                except:
                    print("Could not save figure")
                #plt.show()
                plt.clf()
                fig.clf()
                #fig.close()
                plt.close('all')



        
        #find which opperator the initial state is an eigenvector of
        inits = qt.Qobj(initial_state.full())
        truth = inits.isherm and np.isclose(inits.tr(),1) and np.isclose((inits*inits).tr(),1)
        if not truth:
            print("The initial state is not a pure state!")
        #print("Trace: ",final_state.tr())
        #print(final_state.full())
        eigvals,eigkets = initial_state.eigenstates()
        eigkets,eigvals = [eigket.full() for i,eigket in enumerate(eigkets) if np.isclose(eigvals[i],1)], [eigval for eigval in eigvals if np.isclose(eigval,1)] 
        initial_pure,eigval = eigkets[0],eigvals[0]#!incompatible eigvals
        #print(initial_pure)
        initial_type = None
        fs = final_state.full()
        A = np.dot(initial_pure,initial_pure.conj().T)
        #print(A)
        metric = np.dot(fs,A)
        fidelity = np.trace(metric)
        #fidelity = np.sum([initial_pure[i].conj()*initial_pure[i]*final_state.full()[i,i] for i in range(len(initial_pure))])
        #print("Pure_state: ",initial_pure[:2])
        #print("Fidelity: ",fidelity)
        #prob = np.trace(np.dot(initial_state.full().T.conj(),final_state.full()))#!why are expectation values complex?
        #!why are expectation values complex?
        propabilities.append(fidelity)
    fidelity = np.sum(propabilities)/len(propabilities)
    return fidelity