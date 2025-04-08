import numpy as np
import matplotlib.pyplot as plt
import math
from queue import Queue
import threading
from random import randrange
import os
import statistics
   
case = "coloumb"
verbose = False

hbar = 1
hbarSquare = pow(hbar,2)
m = 1
qe = 1
siliconQ = 14*qe
elekperm = 1
cKonst = 1/(4*math.pi*elekperm)

def float_range(x, y, jump):
    if '.' in str(jump):
        d = len(str(jump).split('.')[1])
    else:
        d = -1
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
numberOfCells = 1
func = np.array([])
funcCords = np.array([])
funcIsCalced = np.array([],dtype='bool')
funcIsNew = np.array([],dtype='bool')
funcIsCalcing = np.array([],dtype='bool')
funcPots = np.array([])
funcVs = np.array([])
stepSize = 0.001#skift pot filen hvis den skiftes
impDec = len(str(stepSize).split('.')[1])
impDec = abs(impDec)+1
cellRad = 10
limitDist = cellRad*numberOfCells
funcPots = np.array([])
coloumbPots = np.array([])
funcPots = np.full(round(2*limitDist/stepSize),0,dtype='float64')

direct = os.getcwd()
if "coloumbFunc.csv" in os.listdir(direct):
    coloumbPots = np.loadtxt(direct + "\\coloumbFunc.csv",delimiter=";") 
else:
    coloumbCap = -5
    coloumbAmm = 100
    if coloumbAmm // 2 == coloumbAmm / 2:
        coloumbAmm = round(coloumbAmm)
        if coloumbAmm // 2 == coloumbAmm / 2:
            coloumbAmm += 1
    coloumbPoints = np.array([])
    coloumbDist = 2 * cellRad
    for x in frange(-coloumbAmm/2*coloumbDist+coloumbDist/2,coloumbAmm/2*coloumbDist+coloumbDist/2,coloumbDist):
        coloumbPoints = np.append(coloumbPoints,x)
    coloumbPots = np.full(len(funcPots)//numberOfCells,0,dtype='float64')
    #for x in frange(-limitDist, limitDist, stepSize):
    #    coloumbPots = np.append(funcPots,0)
    for i,e in enumerate(coloumbPoints):
        print("coloumbPot number " + str(i))
        for i,x in enumerate(frange(-cellRad, cellRad, stepSize)):
            if (r := abs(e+x)) != 0:
                coloumbPots[i] += cKonst * siliconQ * -qe/ r
            else:
                coloumbPots[i] = min(coloumbPots)
    for i,__ in enumerate(coloumbPots):
        if coloumbPots[i] < coloumbCap:
            coloumbPots[i] = coloumbCap
    file = open(direct + "\\coloumbFunc.csv","w+")
    file.close()
    np.savetxt(direct + "\\coloumbFunc.csv",coloumbPots,delimiter=";")

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


if case == "coloumb":
    funcPots = np.tile(coloumbPots,numberOfCells)
elif case == "kronig":
    diff = np.full(coloumbPots.shape[0]-1,0,dtype='float64')
    for i,__ in enumerate(diff):
        diff[i] = (coloumbPots[i+1]-coloumbPots[i])/(funcCords[i+1]-funcCords[i])
    vhChangePoints = np.array([], dtype='int16')
    for i in range(diff.shape[0]-1):
        if (abs(diff[i+1]) <= 1 and abs(diff[i]) >= 1) or (abs(diff[i+1]) >= 1 and abs(diff[i]) <= 1):
            vhChangePoints = np.append(vhChangePoints, i)
    
    vertLines = [0,0]
    vertLines[0] = round(statistics.mean(range(vhChangePoints[0],maxI//2)))
    vertLines[1] = round(statistics.mean(range(maxI//2,vhChangePoints[len(vhChangePoints)-1])))

    horLines = [0,0]
    horLines[0] = statistics.mean(coloumbPots[0:vhChangePoints[0]])
    horLines[1] = statistics.mean(coloumbPots[vhChangePoints[len(vhChangePoints)-1]:maxI])
    horLine = statistics.mean(horLines)

    for j in range(0,numberOfCells):
        for i,__ in enumerate(funcPots):
            funcPots[i+j*len(coloumbPots)] = horLine
        for i in range(vertLines[0],vertLines[1]):
            funcPots[i+j*len(coloumbPots)] = min(coloumbPots)

if verbose:
    plt.plot(funcCords,funcPots)
    plt.plot(funcCords,np.tile(coloumbPots,numberOfCells))
    plt.show() 


funcs = np.array([])

#initializing processing queue
toProces = Queue(maxsize = 3)


finished = False

def waveFuncCalcer():
    while finished == False:
        dataGetSuccesfull = False
        try:
            dat = toProces.get_nowait()
            dataGetSuccesfull = True
        except:
            pass
        if dataGetSuccesfull == True:
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
            funcIsCalcing[indecies[0]+1] = False
            funcIsCalcing[indecies[0]-1] = False
            funcIsNew[indecies[0]] = False
            if indecies[1] != 0 and indecies[1] != maxI:
                funcIsNew[indecies[1]] = True
            funcIsCalced[indecies[1]] = True
        
        
def emptyQueue():
    while toProces.empty == False:
        toProces.get_nowait()
        
            
        
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
        for j,i in enumerate((starts:=np.where(funcIsNew==True)[0])):
            newCords = np.array([])
            if starts.shape[0] > 1:
                if j == 1:
                    iNewCords = [i+1]
                else:
                    iNewCords = [i-1]
                #newCords = [funcCords[iNewCords[0]]]
            elif funcIsCalced[0] == True:
                iNewCords = [i+1]
            elif funcIsCalced[maxI] == True:
                iNewCords = [i-1]
            else:
                iNewCords = [i+1,i-1]
                #newCords = [funcCords[iNewCords[0]],funcCords[iNewCords[1]]]
            for e in iNewCords:
                newCords = np.append(newCords,funcCords[e])
            cords = funcCords[i]
            if np.all(funcIsCalced):
                emptyQueue()
                break
            for i,e in enumerate(newCords):
                if funcIsCalced[iNewCords[i]] == False and funcIsCalcing[iNewCords[i]] == False:
                    funcIsCalcing[iNewCords[i]] = True
                    toProces.put_nowait(np.array((cords,e,E)))

    diffs = [0,0]
    diffs[0] = (func[1]-func[0])/(funcCords[1]-funcCords[0])
    diffs[1] = (func[maxI]-func[maxI-1])/(funcCords[maxI]-funcCords[maxI-1])

    iScore = abs(diffs[1]-diffs[0])
    #iScore = abs(func[0])**3+abs(func[maxI])**3
    EScore = abs(func[0]-func[maxI])
    #EScore = abs(func[0])+abs(func[maxI])
    return iScore, EScore

#initializing energy
E = -2.40
Estep = 0.1
Edir = -1
sizeLimit = 0.001
diffLimit = 0.01
iStepSizeStart = 100
iToTry = maxI//2

finished = False
if __name__ == "__main__":
    EScores = np.array([])
    while True:
        print(E)
        initEnv()
        dirr = 1
        iScores = np.full(func.shape[0],None)
        Escore = 0
        iStepSize = iStepSizeStart
        iToTry -= iStepSize*2
        iTries = np.array([])
        firstITry = True
        while True:
            try:
                if iScores[iToTry] > iScores[iToTry-dirr*iStepSize]:
                    if firstITry == False:
                        iStepSize = iStepSize//2
                    else:
                        firstITry = False
                    dirr = -dirr
            except:
                pass
            iToTry += dirr * iStepSize
            try:
                while iScores[iToTry] != None:
                    iToTry += dirr
            except:
                if iToTry < 0:
                    dirr = abs(dirr)
                    iToTry = max(iTries)
                elif iToTry > maxI:
                    dirr = -abs(dirr)
                    iToTry = min(iTries)
                else:
                    raise Exception("bruh")
            iTries = np.append(iTries,iToTry)
            iScores[iToTry] , Escore = trySolution(iToTry,E)
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
            breakLoop = False
            if len(minVals) >=3:##hvis et minimumspunkt er fundet, er funktionen løst
                for i in range(1,len(minVals)-1):
                    if iScores[minVals[i-1]] >= iScores[minVals[i]] and iScores[minVals[i+1]] >= iScores[minVals[i]]: 
                        breakLoop = True
            if breakLoop:
                break
        print(func[0])
        print(func[maxI])
        

        EScores = np.append(EScores,Escore)
        breakE = False
        if abs(func[maxI]-func[0]) >= sizeLimit:
            if len(EScores) >= 2:
                lastEIndex = len(EScores)-1
                if EScores[lastEIndex] < EScores[lastEIndex-1]:
                    pass
                else:
                    Estep /=2
                    Edir *= -1
        else:
            diffs = [0,0]
            diffs[0] = (func[1]-func[0])/(funcCords[1]-funcCords[0])
            diffs[1] = (func[maxI]-func[maxI-1])/(funcCords[maxI]-funcCords[maxI-1])
            if abs(diffs[1]-diffs[0]) < diffLimit:
                breakE = True
        E += Estep * Edir
        funcsDat = np.append(funcs,[func,E,Escore])
        funcs = np.append(funcs,func)
        if verbose:
            plt.plot(funcCords,funcPots)
            plt.plot(funcCords,func)
            plt.grid(True)
            plt.show() 
        ##fjernet fordi det er lidt tja. det mangler  vist bare at pots skal udvides
        """if abs(func[0]) > abs(2*func[maxI]) or abs(2*func[0]) < abs(func[maxI]):
            stepSize /= 2
            iToTry*=2
            maxI*=2"""
        if breakE:
            break
    

    finished = True
    EPlot = np.full(funcPots.shape,E)
    plt.plot(funcCords,EPlot)
    plt.plot(funcCords,funcPots)
    plt.plot(funcCords,func)
    plt.grid(True)
    plt.show() 