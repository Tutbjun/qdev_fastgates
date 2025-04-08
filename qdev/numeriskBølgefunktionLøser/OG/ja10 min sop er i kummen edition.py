import numpy as np
import matplotlib.pyplot as plt
import math
from queue import Queue
import threading
from random import randrange
import os
import statistics
import time
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize
import functools


#scuffed løsninger, pls fix:
#linje 89
#linje 142

programMode = "tryEK"
case = "coloumb"
verbose = True
verboseLightMode = True
#following variables are ignores if programMode sais so:
kToTry = 1
EToTry = -2
AToTry = 1
BToTry = 1

hbar = 1
hbarSquare = pow(hbar,2)
m = 1
sigma = math.sqrt(2*m)/hbar
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
stepSize = 0.1#skift pot filen hvis den skiftes
impDec = len(str(stepSize).split('.')[1])
impDec = abs(impDec)+1
cellRad = 10
limitDist = cellRad*numberOfCells
#funcPots = np.array([])
coloumbPots = np.array([])
stepCount = int(2*limitDist/stepSize)
if stepCount / 2 == stepCount // 2:
    stepCount += 1
funcPots = np.full(stepCount,0,dtype='float64')

direct = os.getcwd()
if "coloumbFunc.csv" in os.listdir(direct):
    coloumbPots = np.loadtxt(direct + "\\coloumbFunc.csv",delimiter=";") 
else:
    coloumbCap = -3
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
        for j,x in enumerate(frange(-cellRad, cellRad+stepSize, stepSize)):#har bare plusset med stepSize, hvilket er rimelig flippin janktastic
            if (r := abs(e+x)) != 0:
                coloumbPots[j] += cKonst * siliconQ * -qe/ r
            else:
                coloumbPots[j] = min(coloumbPots)
    for i,__ in enumerate(coloumbPots):
        if coloumbPots[i] < coloumbCap:
            coloumbPots[i] = coloumbCap
    file = open(direct + "\\coloumbFunc.csv","w+")
    file.close()
    np.savetxt(direct + "\\coloumbFunc.csv",coloumbPots,delimiter=";")

def initEnv():
    print("initializing enviroment...")
    global func
    global funcCords
    global funcIsCalced
    global funcIsNew
    global funcIsCalcing
    global funcVs
    global funcDiffs
    global funcParams
    funcCords = np.array(frange(-limitDist, limitDist+stepSize, stepSize))
    funcParams = np.full((funcCords.shape[0],2),0,dtype='float64')
    funcDiffs = np.copy(funcParams)
    func = np.copy(funcDiffs)
    funcIsCalced = np.full(funcCords.shape,False)
    funcIsNew = np.copy(funcIsCalced)
    funcIsCalcing = np.copy(funcIsNew)
    funcVs = np.copy(funcParams)

initEnv()
maxI = funcCords.shape[0]-1


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
            vhChangePoints = np.append(vhChangePoints, i+1)
    if len(vhChangePoints) == 0:
        raise Exception("coloumbCap may be too high?") 
    vertLines = [0,0]
    vertLines[0] = round(statistics.mean(range(vhChangePoints[0],vhChangePoints[1]+1)))
    vertLines[1] = round(statistics.mean(range(vhChangePoints[len(vhChangePoints)-2],vhChangePoints[len(vhChangePoints)-1]+4)))#scuffed

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

##file = open(direct + "\\coloumbFunc.csv","w+")
#file.close()
#np.savetxt(direct + "\\coloumbFunc.csv",funcPots,delimiter=";")

if verbose:
    plt.plot(funcCords,funcPots)
    plt.plot(funcCords,np.tile(coloumbPots,numberOfCells))
    if verboseLightMode == False:
        plt.show() 
    else:
        plt.show(block=False)
        try:
            plt.pause(3)
        except:
            pass
        plt.close()

if len(funcPots)!=len(funcCords):
    raise Exception("+1 fejl et sted")

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

def calcFuncVal(x,E,V,k,A,B,tn):
    val = 0
    if E >= V:
        c = math.sqrt(E-V)*sigma
        if A != 0:
            val += A*math.cosh(c*x)*cos(k*x,tn)
        if B != 0:
            val += B*math.sinh(c*x)*sin(k*x,tn)
    elif E < V:
        c = math.sqrt(V-E)*sigma
        if A != 0:
            val += A*cos(c*x,tn)*cos(k*x,tn)
        if B != 0:
            val += B*sin(c*x,tn)*cos(k*x,tn)
    return val

def calcDiffVal(x,E,V,k,A,B,tn):
    val = 0
    if E >= V:
        c = math.sqrt(E-V)*sigma
        if A != 0:
            val += A*c*math.sinh(c*x)*cos(k*x,tn)-A*math.cosh(c*x)*k*math.sin(k*x)
        if B != 0:
            val += B*c*math.cosh(c*x)*sin(k*x,tn) + B*math.sinh(c*x)*k*cos(k*x,tn)
    elif E < V:
        c = math.sqrt(V-E)*sigma
        if A != 0:
            val += -A*c*sin(x*c,tn)*cos(k*x,tn) - A*cos(x*c,tn)*k*sin(k*x,tn)
        if B != 0:
            val += B*c*cos(x*c,tn)*cos(k*x,tn) - B*sin(x*c,tn)*k*sin(k*x,tn)
    return val

def calcB(f,E,V,k,d,x,tn):###mangler
    kx = k*x
    c5 = cos(kx,tn)
    c6 = sin(kx,tn)
    if E >= V:
        c1 = math.sqrt(E-V)
        c2 = sigma*x*c1
        c3 = math.sinh(c2)
        c4 = math.cosh(c2)
        val = -(c1*c3*c5*f*sigma - c4*c6*f*k - d*c4*c5)
        val /= (sigma*c1*c4**2*c6*c5 - c1*c6*c3**2*c5*sigma + c4*c6**2*c3*k + c3*k*c5**2*c4)
    elif E < V:
        c1 = math.sqrt(V-E)
        c2 = sigma*x*c1
        c3 = sin(c2,tn)
        c4 = cos(c2,tn)
        val = c3*c5*c1*f*sigma + c4*c6*f*k + d*c4*c5
        val /= (sigma*c5**2*c1*(c4**2 + c3**2))
    return val

def calcA(f,E,V,k,B,x,tn):###mangler
    kx = k*x
    if E >= V:
        c1 = sigma*x*math.sqrt(E-V)
        val = -B*math.sinh(c1)*sin(kx,tn) + f
        val /= math.cosh(c1)*cos(kx,tn)
    elif E < V:
        c1 = sigma*x*math.sqrt(V - E)
        val = -B*sin(c1,tn)*cos(kx,tn) + f
        val /= cos(c1,tn)*cos(kx,tn)
    return val

calcingSol = False

#plotUndefined=True
#maxPlotVal = 1
def defPlot(maxPlotVal):
    plt.ion()
    fig = plt.figure(figsize=(21, 12))
    axDiff = fig.add_subplot(111)
    axFunc = fig.add_subplot(111)
    axPots = fig.add_subplot(111)
    tempPlot = np.full(funcCords.shape[0],maxPlotVal)
    tempPlot[0] *= -1
    lineDiff, = axDiff.plot(funcCords[0:len(funcCords)],tempPlot, 'r-')
    lineFunc, = axFunc.plot(funcCords[0:len(funcCords)],tempPlot, 'b-')
    linePots, = axFunc.plot(funcCords[0:len(funcCords)],tempPlot, 'g-')
    return fig,lineDiff,lineFunc,linePots

def plot(fig,d,f,lineDiff,lineFunc,linePots,maxPlotVal,plotUndefined):
    lineDiff.set_ydata(d)
    lineFunc.set_ydata(f)

    linePots.set_ydata(funcPots[0:len(funcPots)])
    fig.canvas.draw()

    f = np.absolute(f)
    if max(f) > maxPlotVal:
        maxPlotVal*=3
        plotUndefined = True
        plt.close(fig='all')
    plt.pause(0.01)
    return maxPlotVal, plotUndefined
        


#diffPlotQueue = Queue()
#funcPlotQueue = Queue()
#plotter = threading.Thread(target=plotter, daemon=False)
#plotter.start()

deltaSlope = False

def solPart(dirr,pos,i,E,k,A,B,taylorPrec,prevF,prevD):
    B = calcB(prevF,E,funcPots[i],k,prevD,pos,taylorPrec)
    A = calcA(prevF,E,funcPots[i],k,B,pos,taylorPrec)
    f = calcFuncVal(pos+stepSize*dirr,E,funcPots[i],k,A,B,taylorPrec)
    d = calcDiffVal(pos+stepSize*dirr,E,funcPots[i],k,A,B,taylorPrec)
    return f,d,B,A


def trysol(k,E,A,B):##mangler
    taylorPrec = 6
    if verbose == True:
        print("attempting solution...")
    f = np.full(len(range(0,maxI+1)),0,dtype='float64')
    d = np.copy(f)
    paramAs = np.copy(f)
    paramBs = np.copy(d)

    startPoint = (maxI)//2
    paramAs[startPoint] = A
    paramBs[startPoint] = B
    #c = setCs(E,funcPots[startPoint],k)
    f[startPoint] = calcFuncVal(0,E,funcPots[startPoint],k,A,B,taylorPrec)
    d[startPoint] = calcDiffVal(0,E,funcPots[startPoint],k,A,B,taylorPrec)
    
    plotUndefined = True
    maxPlotVal = 1

    for i in range(1,startPoint+1):
        pos = stepSize*i
        normI = startPoint+i
        f[normI],d[normI],paramBs[normI],paramAs[normI] = solPart(1,pos,normI,E,k,paramAs[normI-1],paramBs[normI-1],taylorPrec,f[normI-1],d[normI-1])
        invI = startPoint-i
        f[invI],d[invI],paramBs[invI],paramAs[invI] = solPart(-1,-pos,invI,E,k,paramAs[(invI)+1],paramBs[(invI)+1],taylorPrec,f[invI+1],d[invI+1])

        if verbose == True:
            if plotUndefined:
                fig,lineDiff,lineFunc,linePots = defPlot(maxPlotVal)
                plotUndefined = False
            maxPlotVal, plotUndefined = plot(fig,paramBs,f,lineDiff,lineFunc,linePots,maxPlotVal,plotUndefined)

    if verboseLightMode == True and verbose == True:
        try:
            plt.pause(5)
        except:
            pass
        plt.close()
    return abs(f[0])+abs(f[len(f)-1]), abs(d[0])+abs(d[len(d)-1])

def evalABdependance(k,E):
    cofStep = 0.05
    maxTestVal = 10
    arraySize = round(maxTestVal/cofStep+1)
    scores = np.full((arraySize,arraySize),0,dtype='float64')
    cofRange = frange(0,maxTestVal+cofStep,cofStep)
    for i,A in enumerate(cofRange):
        print(f"trying with A = {A}")
        for j,B in enumerate(cofRange):
            __,scores[i,j] = trysol(k,E,A,B)

    file = open(direct + "\\ABdepend.csv","w+")
    file.close()
    np.savetxt(direct + "\\ABdepend.csv",scores,delimiter=",", fmt='%f')

    x = cofRange
    y = cofRange
    hf = plt.figure()
    ha = hf.add_subplot(111,projection='3d')
    X, Y = np.meshgrid(x,y)
    ha.plot_surface(X,Y,scores)
    plt.show()

def tryk(k,E):
    #pt ingen AB variation fordi det ser ud til at det er lidt lgegyldigt...
    trysol(k,E,0.1,1)

def tryE(E):
    pass

if __name__ == "__main__":
    if programMode == "evalABdependance":
        evalABdependance(kToTry,EToTry)
    if programMode == "tryEK":
        tryk(kToTry,EToTry)
    if programMode == "tryE":
        tryE(EToTry)