import numpy as np
import matplotlib.pyplot as plt
import math
from queue import Queue
import threading
from random import randrange
import os
import statistics
import time
   

case = "coloumb"
verbose = False

hbar = 1
hbarSquare = pow(hbar,2)
m = 1
sigma = 2*m/hbarSquare
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
func = []
funcCords = np.array([])
funcParams = np.array([])
funcDiffs = np.array([])
funcIsCalced = np.array([],dtype='bool')
funcIsNew = np.array([],dtype='bool')
funcIsCalcing = np.array([],dtype='bool')
funcPots = np.array([])
funcVs = np.array([])
stepSize = 0.05#skift pot filen hvis den skiftes
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
    coloumbCap = -1
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

def initEnv():#############################################fucked tidsforbrug på den her
    print("initializing enviroment...")
    global func
    global funcCords
    global funcIsCalced
    global funcIsNew
    global funcIsCalcing
    global funcVs
    global funcDiffs
    global funcParams
    funcCords = np.array(frange(-limitDist, limitDist, stepSize))
    funcParams = np.full((funcCords.shape[0],2),0,dtype='float64')
    funcDiffs = np.copy(funcParams)
    func = np.copy(funcDiffs)
    funcIsCalced = np.full(funcCords.shape,False)
    funcIsNew = np.copy(funcIsCalced)
    funcIsCalcing = np.copy(funcIsNew)
    funcVs = np.copy(funcParams)

initEnv()
maxI = func.shape[0]-1


if case == "coloumb":
    funcPots = np.tile(coloumbPots,numberOfCells)
elif case == "kronig":
    print("calculating equivalent kronig model...")
    diff = np.full(coloumbPots.shape[0]-1,0,dtype='float64')
    for i,__ in enumerate(diff):
        diff[i] = (coloumbPots[i+1]-coloumbPots[i])/(funcCords[i+1]-funcCords[i])
    vhChangePoints = np.array([], dtype='int16')
    for i,e in enumerate(diff):
        diff[i] = abs(e)
    for i in range(diff.shape[0]-1):
        if (diff[i+1] <= 1 and diff[i] >= 1) or (diff[i+1] >= 1 and diff[i] <= 1):
            vhChangePoints = np.append(vhChangePoints, i)
    if len(vhChangePoints) == 0:
        raise Exception("coloumbCap may be too high?") 
    vertLines = [0,0]
    vertLines[0] = round(statistics.mean(range(vhChangePoints[0],maxI//2)))
    vertLines[1] = round(statistics.mean(range(maxI//2,vhChangePoints[len(vhChangePoints)-1])))

    horLines = [0,0]
    horLines[0] = statistics.mean(coloumbPots[0:vhChangePoints[0]])
    horLines[1] = statistics.mean(coloumbPots[vhChangePoints[len(vhChangePoints)-1]:maxI])
    horLine = statistics.mean(horLines)

    minColoumbPot = min(coloumbPots)
    lenColoumbPot = len(coloumbPots)
    for j in range(0,numberOfCells):
        for i,__ in enumerate(funcPots):
            funcPots[i+j*lenColoumbPot] = horLine
        for i in range(vertLines[0],vertLines[1]):
            funcPots[i+j*lenColoumbPot] = minColoumbPot

if verbose:
    plt.plot(funcCords,funcPots)
    plt.plot(funcCords,np.tile(coloumbPots,numberOfCells))
    plt.show() 


funcs = np.array([])

kRange = [0,2*math.pi]

def cosTaylor(x,n):
    if n == 0:
        return 1
    elif n == 1:
        return -x**2/2
    elif n == 2:
        return x**4/24
    elif n == 3:
        return -x**6/720
    elif n == 4:
        return math.pow(x,8)/40320
    elif n == 5:
        return -math.pow(x,10)/3628800
    elif n == 6:
        return math.pow(x,12)/479001600

halfPi=math.pi/2

def cos(x, n):
    x = abs(x)
    cycles = x/(halfPi)
    x = halfPi*(cycles - int(cycles))
    cycle = int(cycles)+1
    if cycle/2 == cycle//2:
        x-=halfPi
    val = 0
    count = 0
    while (n:=n-1) >= 0:
        val += cosTaylor(x,count)
        count += 1
    wholeCycles = (cycle+2)//2
    if wholeCycles/2 == wholeCycles//2:
        val = -val
    return val

def sin(x, n):
    return cos(x-halfPi,n)

def calcFuncVal(x,c,A,B,tn):
    val = 0
    if A != 0:
        val += A*cos(c[0]*x,tn)
    if B != 0:
        val += B*cos(c[1]*x,tn)
    return val

def calcDiffVal(x,c,A,B,tn):
    val = 0
    if A != 0:
        val+= -c[0]*A*sin(c[0]*x,tn)
    if B != 0:
        val+= -c[1]*B*sin(c[1]*x,tn)
    return val

def calcB(f,d,c1,c2,l,tn):
    val = sin(c1*l,tn)*c1*f+d*cos(c1*l,tn)
    val /= (-c2*sin(c2*l,tn)*cos(c1*l,tn) + cos(c2*l,tn)*sin(c1*l,tn)*c1)
    return val

def calcA(f,B,c1,c2,l,tn):
    val = -B*cos(c2*l,tn)+f
    val /= cos(c1*l,tn)
    return val

def setCs(E,V,k):
    val = sigma*(E-V)
    return [val-k,val+k] 

calcingSol = False

def plotter():
    plt.ion()
    fig = plt.figure()
    axDiff = fig.add_subplot(111)
    axFunc = fig.add_subplot(111)
    axPots = fig.add_subplot(111)
    tempPlot = np.full(funcCords.shape[0]-1,4)
    tempPlot[0] *= -1
    lineDiff, = axDiff.plot(funcCords[0:len(funcCords)-1],tempPlot, 'r-')
    lineFunc, = axFunc.plot(funcCords[0:len(funcCords)-1],tempPlot, 'b-')
    linePots, = axFunc.plot(funcCords[0:len(funcCords)-1],tempPlot, 'g-')
    while True:
        plt.pause(0.01)
        dat = diffPlotQueue.get()
        lineDiff.set_ydata(dat)
        dat = funcPlotQueue.get()
        lineFunc.set_ydata(dat)
        linePots.set_ydata(funcPots[0:len(funcPots)-1])
        fig.canvas.draw()

diffPlotQueue = Queue()
funcPlotQueue = Queue()
plotter = threading.Thread(target=plotter, daemon=False)
plotter.start()

deltaSlope = False

def trysol(k,E,A,B):
    taylorPrec = 6
    print("attempting solution...")
    f = np.full(len(range(0,maxI)),0,dtype='float64')
    d = np.copy(f)
    paramAs = np.copy(f)
    paramBs = np.copy(d)
    paramAs[0] = A
    paramBs[0] = B
    c = setCs(E,funcPots[0],k)
    f[0] = calcFuncVal(0,c,A,B,taylorPrec)
    d[0] = calcDiffVal(0,c,A,B,taylorPrec)
    
    for i in range(1,maxI):####mangler dynamisk taylor n ift. antal bølgelængder
        time.sleep(0.01)
        
        pos = stepSize*i
        
        f[i] = calcFuncVal(pos,c,paramAs[i-1],paramBs[i-1],taylorPrec)
        d[i] = calcDiffVal(pos,c,paramAs[i-1],paramBs[i-1],taylorPrec)

        c = setCs(E,funcPots[i],k)
        
        paramBs[i] = calcB(f[i],d[i],c[0],c[1],pos,taylorPrec)
        paramAs[i] = calcA(f[i],paramBs[i],c[0],c[1],pos,taylorPrec)
        
        
        
        
        #if i > 25:
        #    print("break?")
        

        funcPlotQueue.put_nowait(f)
        diffPlotQueue.put_nowait(paramBs)
        
    calcingSol = False

trysol(0.5,-0.68,0.8,1)

"""

def tryk(k,E):
    pass

def tryE(E):
    global func
    global funcCords
    global funcIsCalced
    global funcIsNew
    global funcIsCalcing
    global funcVs
    print(E)
    initEnv()
    diffsA = np.array([])
    funcA = np.array([])
    k=min(kRange)
    func = np.append(func,calcFuncVal(0,E,k,funcPots[0],funcAParam))
    while k < max(kRange):
        for i in range(1,maxI):
            pass

"""
"""
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
    return iScore, EScore"""
"""
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
        if abs(func[0]) > abs(2*func[maxI]) or abs(2*func[0]) < abs(func[maxI]):
            stepSize /= 2
            iToTry*=2
            maxI*=2
        if breakE:
            break
    

    finished = True
    EPlot = np.full(funcPots.shape,E)
    plt.plot(funcCords,EPlot)
    plt.plot(funcCords,funcPots)
    plt.plot(funcCords,func)
    plt.grid(True)
    plt.show() """