import numpy as np
from numpy import pi
from math import cos, sin, cosh, sinh, sqrt
import matplotlib.pyplot as plt
import sympy as sp
import os
parms = {
    "programMode": "tryAll", 
    "ESpaceSpan": [-1999,100],
    "ESpaceRes": 10, 
    "kSpaceSpan": [0,10],
    "kSpaceRes": 10, 
    "cellRad": 0.5, 
    "cellRes": 101, 
    "atomCharge": 14,
    "electronMass" : 1,
    "potentialCap": -2000,
    "livePlot": True,
    "folder": os.getcwd()

}
constants = {
    "hbar" : 1
}
unit2Natural = {
    "len": 8.400371287*(10**24),
    "charge": 1.60218*(10**-19),
    "mass": 1
}

def varCheck():
    if parms["cellRes"]//2==parms["cellRes"]/2:
        raise ValueError

def genColoumbPot(spaceArea,spaceLen,charge,naturalUnits=True):
    stepSize = (spaceArea[1]-spaceArea[0])/spaceLen
    vals = np.zeros(spaceLen,dtype=float)
    def coloumbFunc(r,q,e):
        k = 1/(4*pi*e)
        v = -q/(r**2)*k
        return v
    vals[0] = -np.inf
    for i in range(1,spaceLen//2):
        vals[i] = coloumbFunc(stepSize*i,charge,1)
        vals[-i] = vals[i]
    return vals
def capColoumb(pot,negativeCap):
    return np.asarray([v if v > negativeCap else negativeCap for v in pot])
def convert2Kronig(pot,negativeCap):
    #TODO: øh ja, hvor sætter man grænserne? I guess der er to graders frihed, men ja?
    pot = capColoumb(pot,negativeCap)
    newNegativeCap = negativeCap/10
    pot = np.asarray([v if v > newNegativeCap else newNegativeCap for v in pot])
    for borderVal in range(len(pot)):
        if pot[borderVal] != newNegativeCap:
            break
    pot[borderVal:] = 0
    pot[-borderVal:] = 0
    return pot

def calcFuncVal(x,E,V,k,A,B,m):#udregner funktionsstørrelsen ved en given x-værdi
    val = 0
    sigma = sqrt(2*m)/constants["hbar"]
    kx = k*x
    c3 = cos(kx)
    c4 = sin(kx)
    if E <= V:#!omvendt af rapporten?!
        c = sqrt(V-E)*sigma
        cx = c*x
        c1 = cosh(cx)
        c2 = sinh(cx)
        if A != 0:
            val += A*complex(c1*c3, -c1*c4)
        if B != 0:
            val += B*complex(c2*c4,c2*c3)
    elif E > V:#!omvendt af rapporten?!
        c = sqrt(E-V)*sigma
        cx = c*x
        c1 = cos(cx)
        c2 = sin(cx)
        if A != 0:
            val += A*complex(c1*c3,-c1*c4)
        if B != 0:
            val += B*complex(c2*c3,-c2*c4)
    return val
def calcDiffVal(x,E,V,k,A,B,m):#udregner differentialestørrelsen ved en given x-værdi
    val = 0
    sigma = sqrt(2*m)/constants["hbar"]
    kx = k*x
    c3 = cos(kx)
    c4 = sin(kx)
    if E <= V:#!omvendt af rapporten?!
        c = sqrt(V-E)*sigma
        cx = c*x
        c1 = cosh(cx)
        c2 = sinh(cx)
        if A != 0:
            val += A*complex(c*c2*c3 - c1*k*c4, - c*c2*c4 - c1*k*c3)
        if B != 0:
            val += B*complex(c*c1*c4 + c2*k*c3,c*c1*c3 - c2*k*c4)
    elif E > V:#!omvendt af rapporten?!
        c = sqrt(E-V)*sigma
        cx = c*x
        c1 = cos(cx)
        c2 = sin(cx)
        if A != 0:
            val += A*complex(-c*c2*c3 - c1*k*c4,c*c2*c4 - c1*k*c3)
        if B != 0:
            val += B*complex(c*c1*c3 - c2*k*c4, - c*c1*c4 - c2*k*c3)
    return val
def calcB(f,E,V,k,d,x,m):#udregner B efter en given formel for at gøre funktionen kontinuerlig
    sigma = sqrt(2*m)/constants["hbar"]
    kx = k*x
    c5 = cos(kx)
    c6 = sin(kx)
    if E <= V:#!omvendt af rapporten?!
        c1 = sqrt(V-E)
        c7 = c1*sigma
        c2 = c7*x
        c3 = sinh(c2)
        c4 = cosh(c2)
        val = complex((-c3*c6*c7*f + c5*c4*f*k + c6*c4*d),(c3*c5*c7*f + c6*c4*f*k - c5*c4*d))
        val /= c7
    elif E > V:#!omvendt af rapporten?!
        c1 = sqrt(E-V)
        c7 = c1*sigma
        c2 = c7*x
        c3 = sin(c2)
        c4 = cos(c2)
        val = complex((c3*c5*c7*f - c4*c6*f*k + c4*c5*d), (c6*c3*c7*f + c4*c5*f*k + c4*c6*d))
        val /= c7
    return val
def calcA(f,E,V,k,d,x,m):#udregner A efter en given formel for at gøre funktionen kontinuerlig
    sigma = sqrt(2*m)/constants["hbar"]
    kx = k*x
    c5 = cos(kx)
    c6 = sin(kx)
    if E <= V:#!omvendt af rapporten?!
        c1 = sqrt(V-E)
        c7 = c1*sigma
        c2 = x*c7
        c3 = sinh(c2)
        c4 = cosh(c2)
        val = complex((c4*c5*c7*f + c6*c3*f*k - c5*c3*d),(c4*c6*c7*f - c5*c3*f*k - c6*c3*d))
        val /= c7
    elif E > V:#!omvendt af rapporten?!
        c1 = sqrt(E-V)
        c7 = c1*sigma
        c2 = x*c7
        c3 = sin(c2)
        c4 = cos(c2)
        val = complex((c6*c3*f*k + c4*c5*c7*f - c3*c5*d),(c6*c4*c7*f - c3*c5*f*k - c6*c3*d))
        val /= c7
    return val

def solPart(step,pos,pot,E,k,A,B,prevf,prevd,mass):#udførrer et Delta x step i løsningen
    B = calcB(prevf,E,pot,k,prevd,pos,mass)
    A = calcA(prevf,E,pot,k,prevd,pos,mass)
    f = calcFuncVal(pos+step,E,pot,k,A,B,mass)
    d = calcDiffVal(pos+step,E,pot,k,A,B,mass)
    return f,d,B,A

def getWaveAt(parms,A0,B0,k,E):#finder en løsningsgraf givet variabler
    pot = parms["pot"]
    f = np.full(parms["cellRes"],0,dtype='complex128')
    d = np.copy(f)
    paramAs = np.copy(f)
    paramBs = np.copy(f)
    x0Index = 0
    paramAs[x0Index] = A0
    paramBs[x0Index] = B0
    f[x0Index] = calcFuncVal(0,E,pot[x0Index],k,A0,B0,parms["electronMass"])
    d[x0Index] = calcDiffVal(0,E,pot[x0Index],k,A0,B0,parms["electronMass"])
    stepSize = parms["cellRad"]*2/parms["cellRes"]
    plotUndefined = True
    for j in range(1,parms["cellRes"]//2):
        pos = stepSize * j
        i = x0Index + j
        f[i],d[i],paramBs[i],paramAs[i] = solPart(
            step=stepSize,
            pos=pos,
            pot=pot[i],
            E=E,
            k=k,
            A=paramAs[i-1],
            B=paramBs[i-1],
            prevf=f[i-1],
            prevd=d[i-1],
            mass=parms["electronMass"]
        )
        invi = x0Index - j
        f[invi],d[invi],paramBs[invi],paramAs[invi] = solPart(
            step=-stepSize,
            pos=-pos,
            pot=pot[invi],
            E=E,
            k=k,
            A=paramAs[invi+1],
            B=paramBs[invi+1],
            prevf=f[invi+1],
            prevd=d[invi+1],
            mass=parms["electronMass"]
        )
        if parms["livePlot"]:
            if plotUndefined:
                plt.close()
                plt.figure(figsize=(12, 10))
                plotUndefined = False
                X = np.linspace(0,parms["cellRes"]//2,parms["cellRes"]//2+1)
                X = np.append(X,-X[1:][::-1])
                fScalar = 0.1*np.max(np.abs(pot))/np.max(np.abs(f))
                plt.plot(X[:len(X)//2],pot[:len(X)//2],c="g")
                plt.plot(X[len(X)//2+1:],pot[len(X)//2+1:],c="g")
                for x,v in zip(X,f):
                    if np.abs(v) > 0:
                        plt.scatter(x,np.real(v)*fScalar,c="r",marker=".")
                        plt.scatter(x,np.imag(v)*fScalar,c="b",marker=".")
                plt.pause(0.05)
            plots = np.array([np.real(f[i]),np.imag(f[i]),np.real(f[invi]),np.imag(f[invi])])
            plots *= fScalar
            plt.scatter(i,plots[0],c="r",marker=".")
            plt.scatter(i,plots[1],c="b",marker=".")
            plt.scatter(invi,plots[2],c="r",marker=".")
            plt.scatter(invi,plots[3],c="b",marker=".")
            plt.pause(0.05)
            if np.any(np.greater(np.abs(plots),np.max(np.abs(pot)))):
                plotUndefined = True
    plt.pause(5)
    plt.close()
    return f,d  

def tryk(parms,k,E):#returnerer funktioner for A og B givet en k-værdi
    fA,dA = getWaveAt(parms,A0=1,B0=0,k=k,E=E)
    fB,dB = getWaveAt(parms,A0=0,B0=1,k=k,E=E)
    return fA,dA,fB,dB

def tryE(E,parms):#finder og returnerer gyldige k-værdier ved en givet E
    pot = parms["pot"]
    if E == 0: E += 0.00000001*(-min(pot))
    kRange = np.linspace(*parms["kSpaceSpan"],parms["kSpaceRes"])
    wA = np.full(len(kRange),0,dtype='complex128')
    wB = np.full(len(kRange),0,dtype='complex128')
    wdA = np.full(len(kRange),0,dtype='complex128')
    wdB = np.full(len(kRange),0,dtype='complex128')
    for i,k in enumerate(kRange):
        print(f"calculating at k={k}")
        fA,dA,fB,dB = tryk(parms,k,E)
        wA[i] = fA[0]-fA[-1]
        wB[i] = fB[0]-fB[-1]
        wdA[i] = dA[0]-dA[-1]
        wdB[i] = dB[0]-dB[-1]
    plt.scatter(kRange,wA,marker=".")
    plt.scatter(kRange,wB,marker=".")
    plt.scatter(kRange,wdA,marker=".")
    plt.scatter(kRange,wdB,marker=".")
    #plt.pause(10)
    fitwA = np.polyfit(kRange, wA, 10)
    fitwB = np.polyfit(kRange, wB, 10)
    fitwdA = np.polyfit(kRange, wdA, 10)
    fitwdB = np.polyfit(kRange, wdB, 10)
    kPltRange = np.linspace(*parms["kSpaceSpan"],parms["kSpaceRes"]*100)
    plt.plot(kRange, np.poly1d(fitwA)(kRange))
    plt.plot(kRange, np.poly1d(fitwB)(kRange))
    plt.plot(kRange, np.poly1d(fitwdA)(kRange))
    plt.plot(kRange, np.poly1d(fitwdB)(kRange))
    plt.pause(2)
    kSym = sp.Symbol('k')
    pol1 = sp.expand(0)
    for i,c in enumerate(fitwA):
        pol1 += sp.expand(c*kSym**i)
    pol2 = sp.expand(0)
    for i,c in enumerate(fitwB):
        pol2 += sp.expand(c*kSym**i)
    pol3 = sp.expand(0)
    for i,c in enumerate(fitwdA):
        pol3 += sp.expand(c*kSym**i)
    pol4 = sp.expand(0)
    for i,c in enumerate(fitwdB):
        pol4 += sp.expand(c*kSym**i)
    M = sp.Matrix([[pol1,pol2],[pol3,pol4]])
    det = M.det()
    maxPower = 0
    try:
        kRoots = sp.Poly(det,kSym).nroots(maxsteps = 100)
    except:
        pass

    """for e in det.args:
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
    kSolves = []"""
    kSolves = [sp.re(e) for e in kRoots if np.abs(sp.re(e))>=100*np.abs(sp.im(e))]
    """try:
        kSolves = mpmath.polyroots(cofs,maxsteps=500)
    except:
        print(f"no sollutions found at E={E}")"""
    return kSolves


def prep(parms):
    pot = genColoumbPot([-parms["cellRad"],parms["cellRad"]],parms["cellRes"],parms["atomCharge"])
    pot = capColoumb(pot,parms["potentialCap"])
    return {"pot": pot}

def saveEkCords(parms,cords):#gemmer resultater
    file = open(parms["folder"] + "/results.csv","w+")
    file.close()
    np.savetxt(parms["folder"] + "/results.csv",cords,delimiter=",", fmt='%f')

def tryAll(parms):#finder helt Ek plot i et angivet interval
    #Estep = (ERange[1]-ERange[0])/parms["ESpaceRes"]
    EsToTry = np.linspace(*parms["ESpaceSpan"],parms["ESpaceRes"])
    Ekcords = np.full((1,2),0,dtype='float64')
    #Ekscores = np.full(1,0,dtype='float64')
    for E in EsToTry:
        print(f"attempting solution with E={E-max(parms['pot'])}")
        validksnScors = np.real(tryE(E,parms))
        for e in validksnScors:
            Ekcords = np.append(Ekcords,[[e,E]], axis=0)
            #Ekscores = np.append(Ekscores,e["score"])
    saveEkCords(parms,Ekcords)
    

if __name__ == "__main__":#starter programmet i forskellige tilstande afhængigt af string variablen
    varCheck()
    preps = prep(parms)
    parms = parms | preps
    if parms["programMode"] == "evalABdependance":
        raise NotImplemented
        evalABdependance(kToTry,EToTry)
    if parms["programMode"] == "tryEk":
        raise NotImplemented
        getWaveAt(kToTry,EToTry,AToTry,BToTry)
    if parms["programMode"] == "tryE":
        raise NotImplemented
        tryE(EToTry)
    if parms["programMode"] == "tryAll":
        tryAll(parms)
