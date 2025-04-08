import numpy as np
import matplotlib.pyplot as plt
import math as m
from queue import Queue
from numba import jit, cuda 
import threading

def float_range(x, y, jump):
    d = len(str(jump).split('.')[1])
    d = abs(d)+1
    if x < y:
        while x < y:
            yield x
            x += jump
            x = round(x,d)
    elif x > y:
        while x > y:
            yield x
            x -= jump
            x = round(x,d)

def frange(start, stop, step):
    return(list(float_range(start,stop,step)))





#inititalizing enviroment
func = np.array([])
funcCords = np.array([])
funcIsCalced = np.array([],dtype='bool')
funcIsNew = np.array([],dtype='bool')
funcIsCalcing = np.array([],dtype='bool')
funcPots = np.array([])
funcVs = np.array([])
stepSize = 0.1
impDec = len(str(stepSize).split('.')[1])
impDec = abs(impDec)+1
limitDist = 10
funcPots = np.array([])
for x in frange(-limitDist, limitDist, stepSize):
    funcPots = np.append(funcPots,0)

def initEnv():
    global func
    global funcCords
    global funcIsCalced
    global funcIsNew
    global funcIsCalcing
    global funcVs
    func = np.array([])
    funcCords = np.array([])
    funcIsCalced = np.array([],dtype='bool')
    funcIsNew = np.array([],dtype='bool')
    funcIsCalcing = np.array([],dtype='bool')    
    funcVs = np.array([])
    for x in frange(-limitDist, limitDist, stepSize):
        funcCords = np.append(funcCords, x)
        funcIsCalced = np.append(funcIsCalced,False)
        funcIsNew = np.append(funcIsNew,False)
        funcIsCalcing = np.append(funcIsCalcing,False)
        funcVs = np.append(funcVs,np.array([0,0]))
        func = np.append(func,0)
    funcVs = np.reshape(funcVs,(funcVs.shape[0]//2,2))

initEnv()
maxI = func.shape[0]-1


funcs = np.array([])


#initializing energy
energyStartRes = 0.1
energyResStepdown = 0.1#vil altså kun køre med en energyres
maxEnergy = 0.2

#initializing processing queue
toProces = Queue(maxsize = 3)

hbar = 1
hbarSquare = pow(hbar,2)
m = 1
finished = False

def waveFuncCalcer():
    while finished == False:
        try:
            dat = toProces.get_nowait()
            cords = dat[0:2]
            E = dat[2]
            indecies = np.array([],dtype='int32')
            for e in cords:
                indecies = np.append(indecies, np.where(funcCords==e))
            pots = np.array([])
            for i in indecies:
                pots = np.append(pots,np.copy(funcPots[i]))
            v = np.copy(funcVs[indecies[0]])
            if (dir:=indecies[1]-indecies[0]) > 0:
                v = v[1]
                dir = 1
            elif dir < 0:
                v = v[0]
                dir = -1
            vIndex = 1
            if dir == -1:
                vIndex = 0
            acc = (pots[0]-E)/hbarSquare * 2 * m * func[indecies[0]]
            funcVs[indecies[1],vIndex] = v + acc * stepSize
            func[indecies[1]] = func[indecies[0]] + (funcVs[indecies[0],vIndex]+funcVs[indecies[1],vIndex])/2*stepSize
            funcIsCalcing[indecies[0]] = False
            funcIsNew[indecies[0]] = False
            if indecies[1] != 0 and indecies[1] != maxI:
                funcIsNew[indecies[1]] = True
            funcIsCalced[indecies[1]] = True
        except:
            pass
        
def emptyQueue():
    while toProces.empty == False:
        toProces.get_nowait()
        
            
        

giveUpLimit = 0.1
fig = plt.figure()
ax = plt.axes(projection='3d')
print(threading.active_count())
Calcer = threading.Thread(target=waveFuncCalcer, daemon=False)
Calcer.start()
threadCount = threading.active_count()

#queueing loop
finished = False
if __name__ == "__main__":
    for Estep in frange(energyStartRes,0,energyResStepdown):
        for E in (energyRange := frange(Estep,maxEnergy,Estep)):
            for iStart in range(int(np.where(funcCords==min(funcCords))[0]),int(np.where(funcCords==max(funcCords))[0])):
                print(iStart)
                emptyQueue()
                initEnv()
                func[iStart] = 1
                funcIsCalced[iStart] = True
                funcIsNew[iStart] = True
                funcLimitIndecies = np.array([0,maxI])
                while np.all(funcIsCalced) == False:
                    for i in np.where(funcIsNew==True)[0]:
                        newCords = np.array([])
                        cords = funcCords[i]
                        newCords = np.append(newCords, cords+stepSize)
                        newCords = np.append(newCords, cords-stepSize)
                        for j,e in enumerate(funcLimitIndecies):
                            if funcIsCalced[e]:
                                if func[e] > giveUpLimit:
                                    funcIsCalced = True
                                    emptyQueue()
                                    break
                                else:
                                    funcLimitIndecies = np.delete(funcLimitIndecies,j)
                                    break
                        if np.all(funcIsCalced):
                            emptyQueue()
                            break
                        for j,e in enumerate(newCords):
                            e = round(e,impDec)
                            if funcIsCalced[(index:=np.where(funcCords==e)[0])] == True:
                                pass
                            elif funcIsCalcing[index] == False:
                                funcIsCalcing[index] = True
                                toProces.put_nowait(np.array((cords,e,E)))
                funcs = np.append(funcs,func)
                funcs = np.reshape(funcs,(funcs.shape[0]//func.shape[0],func.shape[0]))

np.savetxt('datFuncs.csv', funcs, delimiter=';')
np.savetxt('datFunc.csv', func, delimiter=';')

#a = np.loadtxt(url, skiprows=1, delimiter=',')
#a = np.loadtxt(url, skiprows=1, delimiter=',')

finished = True
xTemp = range(0,func.shape[0])
X = np.array([])
for i in range(0, funcs.shape[0]):
    X = np.append(X, xTemp)
X = np.reshape(X,(funcs.shape[0],func.shape[0]))
Y = np.array([])
for i in range(0, funcs.shape[0]):
    Y = np.append(Y, np.full(func.shape[0],i))
Y = np.reshape(Y,(func.shape[0],funcs.shape[0]))
Z = np.copy(funcs)
# syntax for plotting 
ax.plot_surface(X, Y, Z, cmap ='viridis', edgecolor ='green') 
ax.set_title('Surface plot geeks for geeks') 
plt.show() 
            





#foreach newspace
    #find emptyspace
    #if insideBorders and nottaken
        #add to queue
    
#in queue
    #calc nextVal
