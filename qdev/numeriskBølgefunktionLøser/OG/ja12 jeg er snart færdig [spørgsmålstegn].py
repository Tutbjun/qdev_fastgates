import numpy as np
import matplotlib.pyplot as plt
import math
from queue import Queue
from random import randrange
import os
import statistics
import time
from mpl_toolkits.mplot3d import Axes3D
import scipy.optimize
import functools
import keyboard

#scuffed løsninger, pls fix:
#linje 89
#linje 142

programMode = "tryAll"
case = "kronig"
verbose = False
verboseLightMode = False
coloumbCap = -15
stepSize = 0.001#skift pot filen hvis den skiftes
cellRad = 0.5
#following variables are ignores if programMode sais so:
kToTry = 2
EToTry = 9
AToTry = 1
BToTry = 1

hbar = 1
hbarSquare = pow(hbar,2)
m = 1
sigma = math.sqrt(2*m)/hbar
qe = 1.60217662
siliconQ = 14*qe
elekperm = 8.8541878176
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

impDec = len(str(stepSize).split('.')[1])
impDec = abs(impDec)+1

limitDist = cellRad*numberOfCells
#funcPots = np.array([])
coloumbPots = np.array([])
stepCount = int(2*limitDist/stepSize)
if stepCount / 2 == stepCount // 2:
    stepCount += 1
funcPots = np.full(stepCount,0,dtype='float64')

direct = os.getcwd()
if "coloumbFunc.csv" in os.listdir(direct):
    coloumbPots = np.loadtxt(direct + "/coloumbFunc.csv",delimiter=";") 
else:
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
    file = open(direct + "/coloumbFunc.csv","w+")
    file.close()
    np.savetxt(direct + "/coloumbFunc.csv",coloumbPots,delimiter=";")

funcCords = np.array(frange(-limitDist, limitDist+stepSize, stepSize))
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
    slopLim = (max(coloumbPots)-min(coloumbPots))/(max(funcCords)-min(funcCords))
    for i in range(diff.shape[0]-1):
        if (diff[i+1] <= slopLim and diff[i] >= slopLim) or (diff[i+1] >= slopLim and diff[i] <= slopLim):
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

#todo fix det her (måske?):
minPot = min(funcPots)
for i,__ in enumerate(funcPots):
    funcPots[i] -= minPot
    coloumbPots[i] -= minPot

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

def calcFuncVal(x,E,V,k,A,B,tn):#todo find ud af om den super lille komlexe startværdi gør noget
    val = 0
    kx = k*x
    c3 = cos(kx,tn)
    c4 = sin(kx,tn)
    if E >= V:
        c = math.sqrt(E-V)*sigma
        cx = c*x
        c1 = math.cosh(cx)
        c2 = math.sinh(cx)
        if A != 0:
            val += np.complex128(complex(A*c1*c3,-A*c1*c4))
        if B != 0:
            val += np.complex128(complex(B*c2*c4,B*c2*c3))
    elif E < V:
        c = math.sqrt(V-E)*sigma
        cx = c*x
        c1 = cos(cx,tn)
        c2 = sin(cx,tn)
        if A != 0:
            val += np.complex128(complex(A*c1*c3,-A*c1*c4))
        if B != 0:
            val += np.complex128(complex(B*c2*c3,-B*c2*c4))
    return val

def calcDiffVal(x,E,V,k,A,B,tn):#todo find ud af om den super lille komlexe startværdi gør noget
    val = 0
    kx = k*x
    c3 = cos(kx,tn)
    c4 = sin(kx,tn)
    if E >= V:
        c = math.sqrt(E-V)*sigma
        cx = c*x
        c1 = math.cosh(cx)
        c2 = math.sinh(cx)
        if A != 0:
            val += np.complex128(complex(A*c*c2*c3-A*c1*k*c4,-A*c*c2*c4-A*c1*k*c3))
        if B != 0:
            val += np.complex128(complex(B*c*c1*c4 + B*c2*k*c3,B*c*c1*c3-B*c2*k*c4))
    elif E < V:
        c = math.sqrt(V-E)*sigma
        cx = c*x
        c1 = cos(cx,tn)
        c2 = sin(cx,tn)
        if A != 0:
            val += np.complex128(complex(-A*c*c2*c3 - A*c1*k*c4,A*c*c2*c4-A*c1*k*c3))
        if B != 0:
            val += np.complex128(complex(B*c*c1*c3 - B*c2*k*c4,-B*c*c1*c4-B*c2*k*c3))
    return val

def calcB(f,E,V,k,d,x,tn):
    kx = k*x
    c5 = cos(kx,tn)
    c6 = sin(kx,tn)
    if E >= V:
        c1 = math.sqrt(E-V)
        c2 = sigma*x*c1
        c3 = math.sinh(c2)
        c4 = math.cosh(c2)
        val = -c3*c5*c1*f*sigma + c4*(c6*f*k + c5*d)
        val /= c1*c5*c6*sigma + c4*c3*k
    elif E < V:
        c1 = math.sqrt(V-E)
        c2 = sigma*x*c1
        c3 = sin(c2,tn)
        c4 = cos(c2,tn)
        val = c3*c5*c1*f*sigma + c4*c6*f*k + d*c4*c5
        val /= sigma*c1*math.pow(c5,2)
    return val

def calcA(f,E,V,k,d,x,tn):
    kx = k*x
    c5 = cos(kx,tn)
    c6 = sin(kx,tn)
    if E >= V:
        c1 = math.sqrt(E-V)
        c2 = sigma*x*c1
        c3 = math.sinh(c2)
        c4 = math.cosh(c2)
        val = c4*c1*c6*f*sigma + c5*c3*f*k - c3*c6*d
        val /= c1*c5*c6*sigma + c4*c3*k
    elif E < V:
        c1 = math.sqrt(V-E)
        c2 = sigma*x*c1
        c3 = sin(c2,tn)
        val = c5*cos(c2,tn)*c1*f*sigma - c3*c6*f*k - c3*c5*d
        val /= c1*math.pow(c5,2)*sigma
    return val

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
    linePots, = axPots.plot(funcCords[0:len(funcCords)],tempPlot, 'g-')
    return fig,lineDiff,lineFunc,linePots

def plot(fig,d,f,lineDiff,lineFunc,linePots,maxPlotVal,plotUndefined):
    lineDiff.set_ydata(d)
    lineFunc.set_ydata(f)
    global verbose

    linePots.set_ydata(funcPots[0:len(funcPots)])
    try:
        fig.canvas.draw()
    except:
        if keyboard.is_pressed(['ctrl','shift']):
            verbose = False
            time.sleep(1)

    f = np.absolute(f)
    if max(f) > maxPlotVal:
        maxPlotVal*=3
        plotUndefined = True
        plt.close(fig='all')
    plt.pause(0.001)
    return maxPlotVal, plotUndefined

deltaSlope = False

def solPart(dirr,pos,i,E,k,A,B,taylorPrec,prevF,prevD):
    B = calcB(prevF,E,funcPots[i],k,prevD,pos,taylorPrec)
    A = calcA(prevF,E,funcPots[i],k,prevD,pos,taylorPrec)
    f = calcFuncVal(pos+stepSize*dirr,E,funcPots[i],k,A,B,taylorPrec)
    d = calcDiffVal(pos+stepSize*dirr,E,funcPots[i],k,A,B,taylorPrec)
    return f,d,B,A


def trysol(k,E,A,B):#todo imaginær løsning skal med
    global verbose
    taylorPrec = 6
    f = np.full(len(range(0,maxI+1)),0,dtype='complex128')
    d = np.copy(f)
    paramAs = np.copy(f)
    paramAs = paramAs.astype('float64')
    paramBs = np.copy(paramAs)


    startPoint = (maxI)//2
    paramAs[startPoint] = A
    paramBs[startPoint] = B

    f[startPoint] = calcFuncVal(0,E,funcPots[startPoint],k,A,B,taylorPrec)
    d[startPoint] = calcDiffVal(0,E,funcPots[startPoint],k,A,B,taylorPrec)
    
    plotUndefined = True
    maxPlotVal = 1
    
    for i in range(1,startPoint+1):
        pos = stepSize*i
        normI = startPoint+i
        f[normI],d[normI],paramBs[normI],paramAs[normI] = solPart(1,pos,normI,E,k,paramAs[normI-1],paramBs[normI-1],taylorPrec,f[normI-1].real,d[normI-1].real)
        invI = startPoint-i
        f[invI],d[invI],paramBs[invI],paramAs[invI] = solPart(-1,-pos,invI,E,k,paramAs[(invI)+1],paramBs[(invI)+1],taylorPrec,f[invI+1].real,d[invI+1].real)

        if verbose == True:
            if plotUndefined:
                if (maxVal := max(np.absolute(f))) > maxPlotVal: 
                    maxPlotVal = maxVal
                fig,lineDiff,lineFunc,linePots = defPlot(maxPlotVal)
                plotUndefined = False
            imagF = np.full(f.shape,0,dtype='float64')
            for i,__ in enumerate(f):
                imagF[i] = f[i].imag
            maxPlotVal, plotUndefined = plot(fig,imagF,f.astype('float64'),lineDiff,lineFunc,linePots,maxPlotVal,plotUndefined)
        elif keyboard.is_pressed(['ctrl','shift']) and len(plt.get_fignums()) == 0:
            verbose = True
            plotUndefined = True
            time.sleep(1)

    if verboseLightMode == True and verbose == True:
        try:
            plt.pause(5)
        except:
            pass
        plt.close()
    while len(plt.get_fignums())>0:
        try:
            plt.pause(1)
        except:
            pass
    realD = np.full(d.shape,0,dtype='float64')
    for i,__ in enumerate(d):
        realD[i] = d[i].real
    for i,__ in enumerate(d):
        d[i] = d[i].real + d[i].imag
    for i,__ in enumerate(f):
        f[i] = math.sqrt(f[i].real**2 + f[i].imag**2)
    realF = np.full(f.shape,0,dtype='float64')
    for i,__ in enumerate(d):
        realF[i] = f[i]
    return abs(f[0]-f[len(f)-1]) + abs(d[0])+abs(d[len(d)-1]),abs(d[0])+abs(d[len(d)-1]), realF , realD

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

    file = open(direct + "/ABdepend.csv","w+")
    file.close()
    np.savetxt(direct + "/ABdepend.csv",scores,delimiter=",", fmt='%f')

    x = cofRange
    y = cofRange
    hf = plt.figure()
    ha = hf.add_subplot(111,projection='3d')
    X, Y = np.meshgrid(x,y)
    ha.plot_surface(X,Y,scores)
    plt.show()

def tryk(k,E):#todo prøv med B-bølge i starten og så med A-bølge og returner det bedste resultat
    ABToTry = np.full((1,2),0,dtype='float64')
    ABToTry = np.append(ABToTry,[[10,0]],axis=0)
    ABToTry = np.delete(ABToTry,0,axis=0)
    ABToTry = np.append(ABToTry,[[0,10]],axis=0)
    fullScores = np.array([])
    dScores = np.array([])
    funcs = np.full((1,len(funcPots)),0,dtype='float64')
    realDs = np.full((1,len(funcPots)),0,dtype='float64')
    for e in ABToTry:
        val1,val2,val3,val4 = trysol(k,E,e[0],e[1])
        fullScores = np.append(fullScores,val1)
        dScores = np.append(dScores,val2)
        funcs = np.append(funcs,[val3],axis=0)
        realDs = np.append(realDs,[val4],axis=0)
    funcs = np.delete(funcs,0,axis=0)
    realDs = np.delete(realDs,0,axis=0)
    returnIndex = np.argmin(fullScores)
    return fullScores[returnIndex],dScores[returnIndex],funcs[returnIndex],realDs[returnIndex]

def tpCount(d):
    toppunktCount = 0
    for i,__ in enumerate(d[0:len(d)-1]):
        if ((d[i] > 0 and d[i+1] <= 0) or (d[i] < 0 and d[i+1] >= 0)):#erstat 10 med noget klogt
            toppunktCount += 1
    return int(toppunktCount/2)

def tryE(E):
    klist=np.array([],dtype=float)
    kStartStep = 1
    kStep = kStartStep
    toppunktCount = 0
    maxK = math.pi/(2*cellRad)*2
    kStartVal = -maxK
    k = kStartVal+kStartStep
    margen = 0.00001
    lastScores = np.array([])
    findingKs = True
    __,__,__,tempD = tryk(0,E)
    gsToppunktCount = tpCount(tempD)
    __,__,__,tempD = tryk(kStartVal,E)
    lastToppunktCount = tpCount(tempD)
    kMode = -1#kun -1 og 1
    
    while findingKs:
        print(f"trying with k={k}")
        dScore,__, f, d = tryk(k,E)
        f = np.absolute(f)
        toppunktCount = tpCount(d)
        #if kMode == 1:
        if (toppunktCount-lastToppunktCount)*kMode >= 1:
            if dScore < margen*max(f):
                klist = np.append(klist,k)
                print(f"k found at {k}")
                kStep = kStartStep
                lastToppunktCount += 1*kMode
            k -= kStep
            kStep /= 2
            k += kStep
        else:
            if kMode == -1:
                if (k + kStep > 0) == False:
                    pass
                else:
                    kStep = -k/2
            k += kStep
        lastScores = np.insert(lastScores,0,dScore)
        if len(lastScores) >= 2:
            if (lastScores[0] == lastScores[1] and lastScores[1] == lastScores[2]):
                print("k not found )`:")
                kStep = kStartStep
                if round(k,5) == 0.0:
                    kMode = 1
                    lastToppunktCount = gsToppunktCount
                elif kMode == -1:
                    lastToppunktCount -= 1
                elif kMode == 1:
                    lastToppunktCount += 1
                lastScores = np.array([])
        if k > maxK+kStep:
            findingKs = False
    return klist

def tryAll():
    Estep = 0.05
    ERangeRad = 5
    maxE = ERangeRad*max(funcPots)
    minE = -ERangeRad*max(funcPots)
    EsToTry = frange(minE,maxE,Estep)
    Ekcords = np.array([])
    for E in EsToTry:
        print(f"attempting solution with E={E}")
        validks = tryE(E)
        for k in validks:
            Ekcords = np.append(Ekcords,[k,E], axis=0)
    file = open(direct + "/results.csv","w+")
    file.close()
    np.savetxt(direct + "/results.csv",Ekcords,delimiter=",", fmt='%f')
    


if __name__ == "__main__":
    if programMode == "evalABdependance":
        evalABdependance(kToTry,EToTry)
    if programMode == "tryEK":
        tryk(kToTry,EToTry)
    if programMode == "tryE":
        tryE(EToTry)
    if programMode == "tryAll":
        tryAll()
