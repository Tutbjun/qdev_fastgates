import os
def run_jpt(): 
    global IPYNB_FILENAME
    os.system('jupyter nbconvert --execute {:s} --to html'.format(IPYNB_FILENAME))
#run_jpt()

IPYNB_FILENAME = 'gate_prototype.ipynb'
IPYNB_FILENAME = os.path.join(os.path.dirname(__file__), IPYNB_FILENAME)
print(__file__)
import numpy as np
import pickle
from matplotlib import pyplot as plt
def plot_current_perf():
    path = 'fid_logs.pickle'
    path = os.path.join(os.path.dirname(__file__), path)
    with open(path,'rb') as f:
        perf = pickle.load(f)
    for key,entry in perf.items():
        t_g_by_tau = key
        #entry = np.array(entry)**-1
        mean_var = np.mean(entry)
        std_var = np.std(entry)
        if mean_var < 0.1:# and mean_var > 0.078:
            #plt.errorbar(t_g_by_tau, mean_var, yerr=std_var, fmt='.', color='black')
            plt.scatter(t_g_by_tau, mean_var, color='black', marker='.')
    plt.xlabel(r'$t_g/\tau_l$')
    #plt.ylabel(r'$|<\psi_{expected}|\psi_{actual}>|^2$')
    plt.ylabel("Distance fra ønsket i xyz base, lavere er bedre")
    #plt.ylim(0,0.2)
    path = 'fid_vs_tg_by_tau.png'
    path = os.path.join(os.path.dirname(__file__), path)
    #y log
    #plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(path)
    plt.clf()
    print('plotting')
import time
def concatinate_fid_logs():
    path = os.path.dirname(__file__)
    fid_logs = os.listdir(os.path.join(path,'tmp'))
    fid_logs = [os.path.join('tmp',fid) for fid in fid_logs if 'fid_logs_' in fid]
    #wait one second for the file to be written
    time.sleep(1)
    #concatinate the files to one
    path_f = os.path.join(path,'fid_logs.pickle')
    if not os.path.exists(path_f):
        with open(path_f,'wb') as f:
            pickle.dump({},f)
    with open(path_f,'rb') as f:
        current = pickle.load(f)
    for fid in fid_logs:
        fid = os.path.join(path,fid)
        with open(fid,'r') as f:
            addon = f.readline()
        key = complex(addon.split(' [')[0]).real
        value = addon.split(' [')[1]
        value = value.split(']')[0]
        #interpret the value as a literal list
        value = [eval(x) for x in value.split(',')]
        current[key] = value
    with open(path_f,'wb') as f:
        pickle.dump(current,f)
    #remove the files
    for fid in fid_logs:
        os.remove(os.path.join(path,fid))
        

"""for t_g_by_tau in np.linspace(0.5,20,1000):
    import sys,os
    CONFIG_FILENAME = 'gate_prototype_inputs.config'
    CONFIG_FILENAME = os.path.join(os.path.dirname(__file__), CONFIG_FILENAME)
    def main(argv):
        print(argv)
        with open(CONFIG_FILENAME,'w') as f:
            f.write(f"tg_by_tau = {argv}")
        os.system('jupyter nbconvert --execute {:s} --to html'.format(IPYNB_FILENAME))
        return None
    main(t_g_by_tau)
    #plot_current_perf()"""
    
#same thing but with multiprocessing
from multiprocessing import Pool

#t_g_by_tau = [np.linspace(0.01,0.15,33333)] + [np.linspace(0.2,1.5,33333)]
t_g_by_tau = np.logspace(-8,np.log10(20),1000000)
#t_g_by_tau = np.concatenate(t_g_by_tau).flatten()
def run_jpt(itt):
    global IPYNB_FILENAME
    global t_g_by_tau
    tgbytau = t_g_by_tau[itt]
    import sys,os
    CONFIG_FILENAME = 'gate_prototype_inputs.config'
    CONFIG_FILENAME = os.path.join(os.path.dirname(__file__), CONFIG_FILENAME)
    def main(argv):
        print(argv)
        with open(CONFIG_FILENAME,'w') as f:
            f.write(f"tg_by_tau = {argv}")
        os.system('jupyter nbconvert --execute {:s} --to html'.format(IPYNB_FILENAME))
        return None
    main(tgbytau)

    """if itt % 100 == 0:
        concatinate_fid_logs()
        plot_current_perf()"""
    return None

import random
runnerMode = False
if runnerMode:
    pool = Pool(15)
    ins = list(np.arange(len(t_g_by_tau)))
    random.shuffle(ins)
    pool.map(run_jpt, ins)
    pool.close()
    pool.join()

else:
    #every few seconds, concatinate and plot
    while True:
        concatinate_fid_logs()
        plot_current_perf()
        time.sleep(20)
        if len(os.listdir(os.path.dirname(__file__)+'/tmp')) <= 1:
            time.sleep(20)
            if len(os.listdir(os.path.dirname(__file__)+'/tmp')) <= 1:
                break