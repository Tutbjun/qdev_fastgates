import numpy as np
import matplotlib.pyplot as plt
import math as m
from queue import Queue
#from numba import jit, cuda 
import threading
from random import randrange

#todo:
    #startpunkter gentager sig?
    #skal kunne acceptere andre potentialefunktioner
    #skal kunne løse med andre potentialefunktioner
    #skal kunne finde energiniveauer fra 0 til stop

    

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
stepSize = 0.01
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
Calcer1 = threading.Thread(target=waveFuncCalcer, daemon=False)
Calcer1.start()
#Calcer2 = threading.Thread(target=waveFuncCalcer, daemon=False)
#Calcer2.start()

#queueing loop
def trySolution(iStart, E):
    global func
    global funcCords
    global funcIsCalced
    global funcIsNew
    global funcIsCalcing
    global funcVs
    print(iStart)
    emptyQueue()
    initEnv()
    func[iStart] = 1
    funcIsCalced[iStart] = True
    funcIsNew[iStart] = True
    while np.all(funcIsCalced) == False:
        for j,i in enumerate((starts:=np.where(funcIsNew==True)[0])):
            if starts.shape[0] > 1:
                if j == 1:
                    iNewCords = [i+1]
                else:
                    iNewCords = [i-1]
                newCords = [funcCords[iNewCords[0]]]
            else:
                iNewCords = [i+1,i-1]
                newCords = [funcCords[iNewCords[0]],funcCords[iNewCords[1]]]
            cords = funcCords[i]
            if np.all(funcIsCalced):
                emptyQueue()
                break
            for i,e in enumerate(newCords):
                if funcIsCalced[iNewCords[i]] == False and funcIsCalcing[iNewCords[i]] == False:
                    funcIsCalcing[iNewCords[i]] = True
                    toProces.put_nowait(np.array((cords,e,E)))       
    iScore = abs(func[0])**3+abs(func[maxI])**3
    EScore = abs(func[0])+abs(func[maxI])
    return iScore, EScore

#initializing energy
E = 0.3
Estep = 0.05
Edir = 1
funcBorderFoundLimit = 0.01
#lockedI = False
#iToTry = 0

finished = False
if __name__ == "__main__":
    EScores = np.array([])
    while True:
        print(E)
        dirr = 1
        iScores = np.full(func.shape[0],None)
        iStepSize = 10
        #if lockedI == False:
        iToTry = maxI//2-iStepSize
        Escore = 0
        while True:
            try:
                if iScores[iToTry] > iScores[iToTry-dirr*iStepSize]:
                    dirr = -dirr
                    iStepSize = iStepSize//2
            except:
                pass
            #if lockedI == False:
            iToTry += dirr * iStepSize
            while iScores[iToTry] != None:
                iToTry += dirr
            iScores[iToTry] , Escore = trySolution(iToTry,E)
            #if lockedI == True:
                #break
            ##tjekker efter minimum
            minVals = np.array([],dtype='int32')
            for i,e in enumerate(iScores):
                if e != None:
                    minVals = np.append(minVals,i)
            keepList = np.array([], dtype='int16')
            for i,__ in enumerate(minVals):##finder hvilke værdier der er omgivet af to andre
                toTry = True
                try:
                    minVals[i-1]
                    minVals[i+1]
                except:
                    toTry = False
                if toTry and i != 0 and i != len(minVals)-1:
                    if iScores[minVals[i]-1] != None and iScores[minVals[i]+1] != None:
                        if len(np.where(keepList==i-1)[0]) == 0:
                            keepList = np.append(keepList,i-1)
                        if len(np.where(keepList==i)[0]) == 0:
                            keepList = np.append(keepList,i)
                        if len(np.where(keepList==i+1)[0]) == 0:
                            keepList = np.append(keepList,i+1)
            tempMinVals = np.array([], dtype='int16')
            while len(keepList) > 0:##beholder værdier som har data nok til at kunne afgørre et minimumspunkt
                tempMinVals = np.append(tempMinVals,minVals[keepList[0]])
                keepList = np.delete(keepList,0)
            minVals = tempMinVals
            if len(minVals) >=3:##hvis et minimumspunkt er fundet, er funktionen løst
                for i in range(1,len(minVals)-1):
                    if iScores[minVals[i-1]] >= iScores[minVals[i]] and iScores[minVals[i+1]] >= iScores[minVals[i]]: 
                        breakLoop = True
            if breakLoop:
                break
        #lockedI = True
        EScores = np.append(EScores,Escore)
        breakE = False
        if abs(func[len(func)-1]) >= funcBorderFoundLimit or abs(func[0]) >= funcBorderFoundLimit:
            if len(EScores) >= 2:
                lastEIndex = len(EScores)-1
                if EScores[lastEIndex] > EScores[lastEIndex-1]:
                    pass
                else:
                    Estep /=2
                    Edir *= -1
        else:
            breakE = True
        E += Estep * Edir
        funcsDat = np.append(funcs,[func,E,Escore])
        funcs = np.append(funcs,func)
        if breakE:
            break
    

    finished = True

    plt.plot(funcCords,func)
    plt.grid(True)
    plt.show() 