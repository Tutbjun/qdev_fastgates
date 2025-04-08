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
import scipy.linalg
import sympy as sym
import functools
import keyboard
import mpmath

#scuffed løsninger, pls fix:
#linje 89
#linje 142

#todo bro, der skal jo også være im fit på A og B

programMode = "tryEk"
case = "coloumb"
verbose = True
verboseLightMode = False
coloumbCap = -15
stepSize = 0.001#skift pot filen hvis den skiftes
cellRad = 0.5
#following variables are ignores if programMode sais so:
kToTry = 20
EToTry = 15
AToTry = 10
BToTry = 0

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
    return f,d

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

def tryk(k,E):# den skal bare returnere scoreV, scoreH, diffScoreV og diffscoreH for bægge bølger med amplitude lig 1
    fA,dA = trysol(k,E,1,0)
    fB,dB = trysol(k,E,0,1)
    return fA,dA,fB,dB

def tpCount(d):
    toppunktCount = 0
    for i,__ in enumerate(d[0:len(d)-1]):
        if ((d[i] > 0 and d[i+1] <= 0) or (d[i] < 0 and d[i+1] >= 0)):#erstat 10 med noget klogt
            toppunktCount += 1
    return int(toppunktCount/2)

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def tryE(E): #den skal i stedet lave regressioner for uA uB uA' og uB' og så løse ligningsystemet
    kRad = math.pi/cellRad*2
    kStep = 2*kRad / 50
    kRange = frange(-kRad,kRad+kStep,kStep)
    wA = np.full(len(kRange),0,dtype='complex128')
    wB = np.full(len(kRange),0,dtype='complex128')
    wdA = np.full(len(kRange),0,dtype='complex128')
    wdB = np.full(len(kRange),0,dtype='complex128')
    for i,k in enumerate(kRange):
        print(f"calculating at k={k}")
        fA,dA,fB,dB = tryk(k,E)
        wA[i] = fA[0]-fA[len(fA)-1]
        wB[i] = fB[0]-fB[len(fB)-1]
        wdA[i] = dA[0]-dA[len(dA)-1]
        wdB[i] = dB[0]-dB[len(dB)-1]
    
    fitwA = np.polyfit(kRange, wA, 10)
    fitwB = np.polyfit(kRange, wB, 10)
    fitwdA = np.polyfit(kRange, wdA, 10)
    fitwdB = np.polyfit(kRange, wdB, 10)
    #plt.plot(kRange, np.poly1d(fitwA)(kRange))
    #plt.plot(kRange, np.poly1d(fitwB)(kRange))
    #plt.plot(kRange, np.poly1d(fitwdA)(kRange))
    #plt.plot(kRange, np.poly1d(fitwdB)(kRange))
    k = sym.Symbol('k')
    pol1 = sym.expand(0)
    for i,c in enumerate(fitwA):
        pol1 += sym.expand(c*k**i)
    pol2 = sym.expand(0)
    for i,c in enumerate(fitwB):
        pol2 += sym.expand(c*k**i)
    pol3 = sym.expand(0)
    for i,c in enumerate(fitwdA):
        pol3 += sym.expand(c*k**i)
    pol4 = sym.expand(0)
    for i,c in enumerate(fitwdB):
        pol4 += sym.expand(c*k**i)
    M = sym.Matrix([[pol1,pol2],[pol3,pol4]])
    det = M.det()
    maxPower = 0
    for e in det.args:
        for f in e.args:
            if 'k' in str(f):
                comps = str(f).split('*')
                lastComp = comps[len(comps)-1]
                if is_number(lastComp) == False:
                    if 1 > maxPower:
                        maxPower = 1
                elif (val:=int(lastComp)) > maxPower:
                    maxPower = val
    cofs = np.full(maxPower+1,0,dtype='complex128')
    I = mpmath.mp.mpc(0, 1)
    for e in det.args:
        val = mpmath.mpmathify(0)
        power = 0
        if len(e.args) == 0:
            val += mpmath.mpmathify(e)
            power = 0
        elif len(e.args) == 2:
            if str(type(e.args[1])) != "<class 'sympy.core.numbers.ImaginaryUnit'>":
                val += mpmath.mpmathify(e.args[0])
                comps = str(e.args[1]).split('*')
                if comps[-1] != 'k':
                    power = int(comps[-1])
                else:
                    power = 1
            else:
                val += mpmath.mpmathify(e.args[0])*I
                power = 0
        else:
            val += mpmath.mpmathify(e.args[0])*I
            comps = str(e.args[2]).split('*')
            if comps[-1] != 'k':
                power = int(comps[-1])
            else:
                power = 1
        cofs[-power-1] += complex(val)
    kSolves = mpmath.polyroots(cofs,maxsteps=10000)
    tbd = np.array([],dtype='int32')
    for i,__ in enumerate(kSolves):
        kSolves[i] = complex(kSolves[i])
        if abs(kSolves[i].imag) > abs(kSolves[i].real/100):
            tbd = np.append(tbd,i)
    for i in range(len(tbd)-1,-1,-1):
        kSolves = np.delete(kSolves,tbd[i])
    margen = 0.5
    for i in range(len(kSolves)-1,-1,-1):
        print(f"checking solution at k={kSolves[i]}")
        fA,dA,fB,dB = tryk(kSolves[i],E)
        wA = fA[0]-fA[len(fA)-1]
        wB = fB[0]-fB[len(fB)-1]
        wdA = dA[0]-dA[len(dA)-1]
        wdB = dB[0]-dB[len(dB)-1]
        checkScalar = -wA/wB
        rangeRad = margen*(max(np.absolute(fA))+max(np.absolute(fB)))
        rangeRad += complex(0,rangeRad)
        checkVal = wdA + checkScalar*wdB
        if (checkVal < rangeRad and checkVal > -rangeRad) == False:
            kSolves = np.delete(kSolves,i)
    print(kSolves)

    #plt.show()
    return kSolves


def tryAll():
    Estep = 1
    ERangeRad = 3
    maxE = ERangeRad*max(funcPots)
    minE = -ERangeRad*max(funcPots)
    EsToTry = frange(minE,maxE,Estep)
    Ekcords = np.full((1,2),0,dtype='float64')
    for E in EsToTry:
        print(f"attempting solution with E={E}")
        validks = tryE(E)
        for k in validks:
            Ekcords = np.append(Ekcords,[[k.real,E]], axis=0)
    Ekcords = np.delete(Ekcords,0,axis=0)
    Ekcords = np.delete(Ekcords,0)
    file = open(direct + "/results.csv","w+")
    file.close()
    np.savetxt(direct + "/results.csv",Ekcords,delimiter=",", fmt='%f')
    


if __name__ == "__main__":
    if programMode == "evalABdependance":
        evalABdependance(kToTry,EToTry)
    if programMode == "tryEk":
        trysol(kToTry,EToTry,AToTry,BToTry)
    if programMode == "tryE":
        tryE(EToTry)
    if programMode == "tryAll":
        tryAll()
