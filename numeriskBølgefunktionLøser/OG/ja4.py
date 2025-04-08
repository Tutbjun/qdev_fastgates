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
Calcer2 = threading.Thread(target=waveFuncCalcer, daemon=False)
Calcer2.start()

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
        for i in np.where(funcIsNew==True)[0]:
            newCords = [round(funcCords[i]+stepSize,impDec),round(funcCords[i]-stepSize,impDec)]
            iNewCords = [np.where(funcCords==newCords[0]),np.where(funcCords==newCords[1])]
            cords = funcCords[i]
            #newCords = np.append(newCords, cords+stepSize)
            #newCords = np.append(newCords, cords-stepSize)
            if np.all(funcIsCalced):
                emptyQueue()
                break
            for i,e in enumerate(newCords):
                index = iNewCords[i]
                try:
                    index = index[0][0]
                except:
                    pass
                if funcIsCalced[index] == False and funcIsCalcing[index] == False:
                    funcIsCalcing[index] = True
                    toProces.put_nowait(np.array((cords,e,E)))       
    #calcScore
    iScore = abs(func[0])**3+abs(func[maxI])**3
    return iScore, 0
    #funcs = np.append(funcs,func)
    #funcs = np.reshape(funcs,(funcs.shape[0]//func.shape[0],func.shape[0]))

#initializing energy
E = 0.3

finished = False
if __name__ == "__main__":
    iToTry = maxI//2
    dirr = 1
    iScores = np.full(func.shape[0],None)
    iStepSize = 10
    while True:
        try:
            if iScores[iToTry] > iScores[iToTry-dirr*iStepSize]:
                dirr = -dirr
                iStepSize -=1
        except:
            pass
        #while iScores[iToTry] != None:
        iToTry += dirr * iStepSize
        iScores[iToTry] , Escore = trySolution(iToTry,E)

        minVals = np.array([],dtype='int32')
        for i,e in enumerate(iScores):
            if e != None:
                minVals = np.append(minVals,i)
        tempMinVals = np.copy(minVals)
        for i,e in enumerate(tempMinVals):
            if i-1 >= 0:
                laste = tempMinVals[i-1]
                if iScores[e] < iScores[laste]:
                    minVals = np.delete(minVals,np.where(minVals<=e-1)[0])
                elif iScores[e] > iScores[laste]:
                    minVals = np.delete(minVals,np.where(minVals==e)[0])
        if iScores[min(minVals)-1] != None and iScores[max(minVals)+1] != None:
            if iScores[min(minVals)-1] > iScores[min(minVals)] and iScores[max(minVals)+1] > iScores[max(minVals)]:
                break
        
        """
        try:
            minIs = np.where((iScores != None and iScores == np.amin(iScores)))
        except:
            pass"""
        """
        whereNotNone = np.where(iScores!=None)[0]
        notNoneScoreIs = np.where(iScores[whereNotNone]==min(iScores[whereNotNone]))[0]
        for i in range(0,notNoneScoreIs.shape[0]):
            notNoneScoreIs[i] += min(whereNotNone)
        minVals = np.array(np.where(iScores==min(iScores[whereNotNone]))[0], dtype='int32')
        if iScores[min(minVals)-1] != None and iScores[max(minVals)+1] != None:
            if iScores[min(minVals)-1] > iScores[min(minVals)] and iScores[max(minVals)+1] > iScores[max(minVals)]:
                break"""
    funcs = np.append(funcs,func)


finished = True

plt.plot(funcCords,func)
plt.grid(True)
plt.show() 