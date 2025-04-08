#assuming josephson junction and natural units
#width = 1e-1
#height = 1
#fidelity = 1e-6

#pot = np.arange(0, 1, fidelity)

#define some numerical potential to solve the wavefunction for

import numpy as np
import matplotlib.pyplot as plt
def sinpot(fidelity, k=1, spacings=None):
    if spacings is None:
        X = np.arange(0, 1, fidelity)
    else:
        X = spacings
    Y = np.sin(2*np.pi*X*k)
    dX = X[1:]-X[:-1]
    return dX, X, Y

pot = sinpot(1e-6,k=10)
#plt.plot(pot[1], pot[2])
#plt.savefig('plot.png')
#plt.show()


#define a wavefunction
#for now just a complex wave
def wavefunc_euler(x, k, phase):
    return np.exp(1j*(k*x))*phase

wave = wavefunc_euler

#select a value at a point, with some k and some phase



#def shrodinger_eq(pot, E

def step_bondarry_solver(E,V,width,oldval,dir=1,m=1,hbar=1,kglobal=0):
    if E-V > 0:
        k = np.sqrt(2*m*(E-V))/hbar
    elif E-V < 0:
        k = 1j*np.sqrt(2*m*(V-E))/hbar
    else:
        k = 0
    k -= kglobal
    #fit a phase such that the new wave will have such value at x=0
    phasefit = oldval
    #print(phasefit)
    #evolve the wavefunction with the given width
    val = wave(width*dir, k, phasefit)
    return val

#for now do an example
for E in np.linspace(-5,1.5,100):
    print(E)
    #E = -5
    x = 0.5
    k_lattice = 10
    psi = np.zeros(len(pot[0])+1, dtype=complex)
    psi[0] = 1
    N = len(pot[0])
    from tqdm import tqdm
    """for i in tqdm(range(N)):
        psi[i+1] = step_bondarry_solver(E, pot[2][i]*5, pot[0][i], psi[i], dir=1, kglobal=k_lattice)

    plt.plot(pot[1][:N+1], np.real(psi))
    plt.plot(pot[1][:N+1], pot[2][:N+1])
    plt.savefig('plot.png')
    plt.show()"""

    #now for some fun... given a lattice k, locate a starting phase that comes closest to a continuous wavefunction at the 0,1 boundary

    def eval_psi(E, k_lattice, pot,phase=1, progress=True, dir=1):
        psi = np.zeros(len(pot[0])+1, dtype=complex)
        N = len(pot[0])
        _tqdm_ = tqdm if progress else lambda x: x
        if dir > 0:
            psi[0] = phase
            for i in _tqdm_(range(N)):
                psi[i+1] = step_bondarry_solver(E, pot[2][i], pot[0][i], psi[i], dir=dir, kglobal=k_lattice)
        else:
            psi[N] = phase
            for i in _tqdm_(range(N)):
                psi[N-i-1] = step_bondarry_solver(E, pot[2][N-i-1], pot[0][N-i-1], psi[N-i], dir=dir, kglobal=k_lattice)
        return psi

    #psi_real = eval_psi(E, k_lattice, pot, phase=1)
    #psi_imag = eval_psi(E, k_lattice, pot, phase=1j)
    #print(np.abs(psi_real[-1]))
    #print(np.abs(psi_imag[-1]))

    #starting phase does probably nothing... maybe evaluate phase-difference instead?
    itt = 100
    phase_delta = np.zeros(itt, dtype=complex)
    k_space = np.logspace(-1, 3, itt)
    def task(i, k_lattice):
        i, k_lattice = i
        psi = np.zeros(len(pot[0])+1, dtype=complex)
        #psi = eval_psi(E, k_lattice, pot, phase=1,progress=False)
        psi[len(pot[0])//2] = 1
        pot_right = [pot[0][len(pot[0])//2:], pot[1][len(pot[0])//2:], pot[2][len(pot[0])//2:]]
        psi[len(pot[0])//2:] = eval_psi(E, k_lattice, pot_right, phase=1, progress=False, dir=1)
        pot_left = [pot[0][:len(pot[0])//2+1], pot[1][:len(pot[0])//2+1], pot[2][:len(pot[0])//2+1]]
        psi[:len(pot[0])//2+2] = eval_psi(E, k_lattice, pot_left, phase=1, progress=False, dir=-1)
        fig, ax = plt.subplots(1,1)
        plt.plot(pot[1][:N+1], np.real(psi))
        plt.plot(pot[1][:N+1], pot[2][:N+1])
        plt.savefig('plot_wave.png')
        plt.show()
        #calc differential value
        diff  = psi[1:]-psi[:-1]
        diff = diff/np.abs(diff)

        #do a phaseplot of the start and end of wavefunction
        fig, ax = plt.subplots(1,1)
        ax.plot(np.real(psi[0]), np.imag(psi[0]), 'gx')
        ax.plot(np.real(psi[-1]), np.imag(psi[-1]), 'gx')
        #ax.plot(np.real(diff[0]), np.imag(diff[0]), 'rx')
        #ax.plot(np.real(diff[-1]), np.imag(diff[-1]), 'rx')
        ax.plot(np.real(psi), np.imag(psi), linewidth=0.5, color='green')
        #ax.plot(np.real(diff), np.imag(diff), linewidth=0.5, color='red')
        #set x and y limits to be the same
        ax.set_xlim([-1.1,1.1])
        ax.set_ylim([-1.1,1.1])
        plt.savefig(f'plot_phasewave_{i}.png')
        phase_delta[i] = psi[-1]-psi[0]
        plt.show()
        return psi[-1]-psi[0]

    if False:
        for i,k_lattice in tqdm(enumerate(k_space)):
            task(i, k_lattice)
    else:#do the same but multithreaded
        from multiprocessing import Pool
        from functools import partial
        p = Pool(16)
        args = [(i, k_lattice) for i,k_lattice in enumerate(k_space)]
        result = p.map(partial(task, k_lattice=k_lattice), args)
        phase_delta = np.array(result)
        p.close()
        p.join()
        

    fig, ax = plt.subplots(1,1)
    ax.plot(np.real(phase_delta), np.imag(phase_delta), 'x')
    plt.savefig('plot_delta.png')
    plt.show()


    #stitch the saved phasewaves to an mp4
    import os
    import PIL

    files = [f for f in os.listdir('.') if 'plot_phasewave' in f]
    #files.sort()
    files = sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    images = []
    for f in files:
        images.append(PIL.Image.open(f))

    images[0].save(f'phasewave_E={E}.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
