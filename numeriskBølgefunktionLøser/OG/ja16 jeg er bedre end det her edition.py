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
fileP = os.getcwd()

programMode = "tryAll"
case = "kronig"
verbose = True
verboseLightMode = False
showkPlots = True
coloumbCap = -45
stepSize = 0.01#skift pot filen hvis den skiftes
cellRad = 0.5

useKeyboardLib = False
constantScaler = 0.01

kToTry = 0.4
EToTry = -1*11.19810654216253
AToTry = 10
BToTry = 10

hbar = 1
hbarSquare = pow(hbar,2)
m = 1
sigma = math.sqrt(2*m)/hbar
qe = 1
siliconQ = 14*qe
elekperm = 1
cKonst = 1/(4*math.pi*elekperm)

def float_range(x, y, jump):#utility funktion til at lave en floating point range
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

def frange(start, stop, step):#funktionen der kaldes til en floating point range
    return(list(float_range(start,stop,step)))

numberOfCells = 1

funcPots = np.array([])

impDec = len(str(stepSize).split('.')[1])
impDec = abs(impDec)+1

limitDist = cellRad*numberOfCells
coloumbPots = np.array([])
stepCount = int(2*limitDist/stepSize)
if stepCount / 2 == stepCount // 2:
    stepCount += 1
funcPots = np.full(stepCount,0,dtype='float64')

#generering af potentialer:
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
        for j,x in enumerate(frange(-cellRad, cellRad+stepSize, stepSize)):
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
    vertLines[1] = round(statistics.mean(range(vhChangePoints[len(vhChangePoints)-2],vhChangePoints[len(vhChangePoints)-1]+4)))

    horLines = [0,0]
    horLines[0] = max(coloumbPots)#statistics.mean(coloumbPots[0:vhChangePoints[0]])
    horLines[1] = max(coloumbPots)#statistics.mean(coloumbPots[vhChangePoints[len(vhChangePoints)-1]:maxI])
    horLine = statistics.mean(horLines)

    minColoumbPot = min(coloumbPots)
    lenColoumbPot = len(coloumbPots)
    for j in range(0,numberOfCells):
        for i,__ in enumerate(funcPots):
            funcPots[i+j*lenColoumbPot] = horLine
        for i in range(vertLines[0],vertLines[1]):
            funcPots[i+j*lenColoumbPot] = minColoumbPot

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

def cosTaylor(x,n):#taylorpolynomiet for cos
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

def cos(x, n):#en taylor polynomie implementering af cos bare for sjov
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

def sin(x, n):#en taylor polynomie implementering af sin bare for sjov
    return cos(x-halfPi,n)

def calcFuncVal(x,E,V,k,A,B,tn):#udregner funktionsstørrelsen ved en given x-værdi
    global sigma
    val = 0
    kx = k*x
    c3 = cos(kx,tn)
    c4 = sin(kx,tn)
    if E < V:
        c = math.sqrt(V-E)*sigma
        cx = c*x
        c1 = math.cosh(cx)
        c2 = math.sinh(cx)
        if A != 0:
            val += A*complex( c1*c3, -c1*c4)
        if B != 0:
            val += B*complex( c2*c4,  c2*c3)
    elif E >= V:
        c = math.sqrt(E-V)*sigma
        cx = c*x
        c1 = cos(cx,tn)
        c2 = sin(cx,tn)
        if A != 0:
            val += A*complex( c1*c3, -c1*c4)
        if B != 0:
            val += B*complex( c2*c3, -c2*c4)
    return val

def calcDiffVal(x,E,V,k,A,B,tn):#udregner differentialestørrelsen ved en given x-værdi
    global sigma
    val = 0
    kx = k*x
    c3 = cos(kx,tn)
    c4 = sin(kx,tn)
    if E < V:
        c = math.sqrt(V-E)*sigma
        cx = c*x
        c1 = math.cosh(cx)
        c2 = math.sinh(cx)
        if A != 0:
            val += A*complex( c*c2*c3 -c1*k*c4, -c*c2*c4 -c1*k*c3)
        if B != 0:
            val += B*complex( c*c1*c4 +c2*k*c3,  c*c1*c3 -c2*k*c4)
    elif E >= V:
        c = math.sqrt(E-V)*sigma
        cx = c*x
        c1 = cos(cx,tn)
        c2 = sin(cx,tn)
        if A != 0:
            val += A*complex(-c*c2*c3 -c1*k*c4,  c*c2*c4 -c1*k*c3)
        if B != 0:
            val += B*complex( c*c1*c3 -c2*k*c4, -c*c1*c4 -c2*k*c3)
    return val

def calcB(f,E,V,k,d,x,tn):#udregner B efter en given formel for at gøre funktionen kontinuerlig
    global sigma
    kx = k*x
    c5 = cos(kx,tn)
    
    c6 = sin(kx,tn)
    if E < V:
        c1 = math.sqrt(V-E)
        c7 = c1*sigma
        c2 = c7*x
        c3 = math.cosh(c2)
        c4 = math.sinh(c2)
        val = complex((-c4*c6*c7*f + c5*c3*f*k + c6*c3*d),(c4*c5*c7*f + c6*c3*f*k - c5*c3*d))
        val /= c7
    elif E >= V:
        c1 = math.sqrt(E-V)
        c7 = c1*sigma
        c2 = c7*x
        c3 = cos(c2,tn)
        c4 = sin(c2,tn)
        val = complex((c4*c5*c7*f - c3*c6*f*k + c3*c5*d), (c6*c4*c7*f + c3*c5*f*k + c3*c6*d))
        val /= c7
    return val

def calcA(f,E,V,k,d,x,tn):#udregner A efter en given formel for at gøre funktionen kontinuerlig
    global sigma
    kx = k*x
    c5 = cos(kx,tn)
    c6 = sin(kx,tn)
    if E < V:
        c1 = math.sqrt(V-E)
        c7 = c1*sigma
        c2 = x*c7
        c3 = math.cosh(c2)
        c4 = math.sinh(c2)
        val = complex((c3*c5*c7*f + c6*c4*f*k - c5*c4*d),(c3*c6*c7*f - c5*c4*f*k - c6*c4*d))
        val /= c7
    elif E >= V:
        c1 = math.sqrt(E-V)
        c7 = c1*sigma
        c2 = x*c7
        c3 = cos(c2,tn)
        c4 = sin(c2,tn)
        val = complex((c6*c4*f*k + c3*c5*c7*f - c4*c5*d),(c6*c3*c7*f - c4*c5*f*k - c6*c4*d))
        val /= c7
    return val

def defPlot(maxPlotVal):#sætter plotting variabler op til liveplot
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

def plot(fig,d,f,lineDiff,lineFunc,linePots,maxPlotVal,plotUndefined):#opdaterer liveplot
    lineDiff.set_ydata(d)
    lineFunc.set_ydata(f)
    linePots.set_ydata(funcPots[0:len(funcPots)]/constantScaler)
    try:
        fig.canvas.draw()
    except:
        if keyboard.is_pressed(['ctrl','shift']):
            verbose = False
            time.sleep(1)
    f = np.absolute(f)
    d = np.absolute(d)
    if max(f) > maxPlotVal or max(d) > maxPlotVal:
        maxPlotVal*=3
        plotUndefined = True
        plt.close(fig='all')
    plt.pause(0.001)
    return maxPlotVal, plotUndefined

deltaSlope = False

def solPart(dirr,pos,i,E,k,A,B,taylorPrec,prevF,prevD):#udførrer et Delta x step i løsningen
    B = calcB(prevF,E,funcPots[i],k,prevD,pos,taylorPrec)
    A = calcA(prevF,E,funcPots[i],k,prevD,pos,taylorPrec)
    f = calcFuncVal(pos+stepSize*dirr,E,funcPots[i],k,A,B,taylorPrec)
    d = calcDiffVal(pos+stepSize*dirr,E,funcPots[i],k,A,B,taylorPrec)
    return f,d,B,A

def trysol(k,E,A0,B0):#finder en løsningsgraf givet variabler
    global verbose
    taylorPrec = 6
    f = np.full(len(range(0,maxI+1)),0,dtype='complex128')
    d = np.copy(f)
    paramAs = np.copy(f)
    paramBs = np.copy(paramAs)
    startPoint = (maxI)//2
    paramAs[startPoint] = A0
    paramBs[startPoint] = B0
    f[startPoint] = calcFuncVal(0,E,funcPots[startPoint],k,A0,B0,taylorPrec)
    d[startPoint] = calcDiffVal(0,E,funcPots[startPoint],k,A0,B0,taylorPrec)
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
                if (maxVal := max(np.absolute(f))) > maxPlotVal: 
                    maxPlotVal = maxVal
                fig,lineDiff,lineFunc,linePots = defPlot(maxPlotVal)
                plotUndefined = False
            imagF = np.full(f.shape,0,dtype='float64')
            for i,__ in enumerate(f):
                imagF[i] = f[i].imag
            maxPlotVal, plotUndefined = plot(fig,imagF,f.astype('float64'),lineDiff,lineFunc,linePots,maxPlotVal,plotUndefined)
        elif useKeyboardLib == True:
            if keyboard.is_pressed(['ctrl','shift']) and len(plt.get_fignums()) == 0:
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

def evalABdependance(k,E):#en funktion til automatisk at afprøve start A og B værdier (bliver ikke brugt i den hvoedsagelige løsning)
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

def tryk(k,E):#returnerer funktioner for A og B givet en k-værdi
    fA,dA = trysol(k,E,1,0)
    fB,dB = trysol(k,E,0,1)
    return fA,dA,fB,dB

def tpCount(d):#tæller antal toppunkter ved input af differentialfunktionen (ikke brugt i øjeblikket)
    toppunktCount = 0
    for i,__ in enumerate(d[0:len(d)-1]):
        if ((d[i] > 0 and d[i+1] <= 0) or (d[i] < 0 and d[i+1] >= 0)):
            toppunktCount += 1
    return int(toppunktCount/2)

def is_number(s):#er en string et tal?
    try:
        float(s)
        return True
    except ValueError:
        return False

def tryE(E):#finder og returnerer gyldige k-værdier ved en givet E
    if 0 == E or E in funcPots:
        print("warning, divission with zero will most likely happen. changing E variable...")
        E += 0.00000001
    kRad = math.pi/cellRad*2
    kStep = 2*kRad / 200
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
    bru1 = np.array(kRange)
    bru2 = np.real(wA)
    bru3 = np.imag(wA)
    tjaz = np.stack((np.array(kRange),np.real(wA),np.imag(wA)),axis=-1)
    tempF = open("wA.csv", 'w')
    tempF = open("wB.csv", 'w')
    tempF = open("wdA.csv", 'w')
    tempF = open("wdB.csv", 'w')
    tempF.close()
    np.savetxt(fileP + "/wA.csv",np.stack((np.array(kRange),np.real(wA),np.imag(wA)),axis=-1),delimiter=";")
    np.savetxt(fileP + "/wB.csv",np.stack((np.array(kRange),np.real(wB),np.imag(wB)),axis=-1),delimiter=";")
    np.savetxt(fileP + "/wdA.csv",np.stack((np.array(kRange),np.real(wdA),np.imag(wdA)),axis=-1),delimiter=";")
    np.savetxt(fileP + "/wdB.csv",np.stack((np.array(kRange),np.real(wdB),np.imag(wdB)),axis=-1),delimiter=";",)
    fitwA = np.polyfit(kRange, wA, 10)
    fitwB = np.polyfit(kRange, wB, 10)
    fitwdA = np.polyfit(kRange, wdA, 10)
    fitwdB = np.polyfit(kRange, wdB, 10)
    kRange = frange(-kRad,kRad+kStep,kStep/100)
    if verbose == True or showkPlots == True:
        fit=np.real(np.poly1d(fitwA)(kRange))
        ehm = np.abs(fit)
        maxV=max(np.abs(fit))
        plt.plot(kRange, fit/maxV)
        fit=np.real(np.poly1d(fitwB)(kRange))
        ehm = np.abs(fit)
        maxV=max(np.abs(fit))
        plt.plot(kRange, fit/maxV)
        fit=np.real(np.poly1d(fitwdA)(kRange))
        ehm = np.abs(fit)
        maxV=max(np.abs(fit))
        plt.plot(kRange, fit/maxV)
        fit=np.real(np.poly1d(fitwdB)(kRange))
        ehm = np.abs(fit)
        maxV=max(np.abs(fit))
        plt.plot(kRange, fit/maxV)
        plt.show()
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
    kSolves = []
    try:
        kSolves = mpmath.polyroots(cofs,maxsteps=500)
    except:
        print(f"no sollutions found at E={E}")
    tbd = np.array([],dtype='int32')
    for i,__ in enumerate(kSolves):
        kSolves[i] = complex(kSolves[i])
        if abs(kSolves[i].imag) > abs(kSolves[i].real/100) or abs(kSolves[i]) > max(kRange) or kSolves[i] != kSolves[i]:
            tbd = np.append(tbd,i)
    for i in range(len(tbd)-1,-1,-1):
        kSolves = np.delete(kSolves,tbd[i])
    ksnScors = np.full((len(kSolves),2),0,dtype='complex128')
    if len(kSolves) > 0:
        ksnScors[0:ksnScors.shape[0],0] = kSolves
        for i,__ in enumerate(ksnScors):
            print(f"checking solution at k={kSolves[i]}")
            fA,dA,fB,dB = tryk(kSolves[i].real,E)
            wA = fA[0]-fA[len(fA)-1]
            wB = fB[0]-fB[len(fB)-1]
            wdA = dA[0]-dA[len(dA)-1]
            wdB = dB[0]-dB[len(dB)-1]
            checkScalar = -wA/wB
            checkVal = wdA + checkScalar*wdB
            ksnScors[i,1] = abs(checkVal)/math.sqrt(max(np.absolute(fA))+max(np.absolute(fB)))
    return ksnScors

def saveEkCords(cords,scores):#gemmer resultater
    file = open(direct + "/results.csv","w+")
    file.close()
    np.savetxt(direct + "/results.csv",cords,delimiter=",", fmt='%f')
    file = open(direct + "/resultScores.csv","w+")
    file.close()
    np.savetxt(direct + "/resultScores.csv",scores, fmt='%f')

def tryAll():#finder helt Ek plot i et angivet interval
    ERangeRad = 0.5
    maxE = 1*max(funcPots)
    minE = 0.5*max(funcPots)
    Estep = (maxE-minE)/200
    EsToTry = frange(minE,maxE,Estep)
    Ekcords = np.full((1,2),0,dtype='float64')
    Ekscores = np.full(1,0,dtype='float64')
    for E in EsToTry:
        print(f"attempting solution with E={E/max(funcPots)}")
        validksnScors = np.real(tryE(E))
        for e in validksnScors:
            Ekcords = np.append(Ekcords,[[e[0],E/max(funcPots)]], axis=0)
            Ekscores = np.append(Ekscores,e[1])
        saveEkCords(Ekcords,Ekscores)
    Ekcords = np.delete(Ekcords,0,axis=0)
    Ekscores = np.delete(Ekscores,0)
    saveEkCords(Ekcords,Ekscores)

if __name__ == "__main__":#starter programmet i forskellige tilstande afhængigt af string variablen
    if programMode == "evalABdependance":
        evalABdependance(kToTry,EToTry)
    if programMode == "tryEk":
        trysol(kToTry,EToTry,AToTry,BToTry)
    if programMode == "tryE":
        tryE(EToTry)
    if programMode == "tryAll":
        tryAll()
