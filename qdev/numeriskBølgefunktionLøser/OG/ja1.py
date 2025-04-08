import numpy as np
import matplotlib.pyplot as plt
import math as m
import random as rand

unitSize = 0.1
potStepSize = 1
pot = np.array([2**100,2**100,0,2**100,2**100])

randFunc = 0.0001
func = np.array([0])
hBar = 1
hBarSquare = m.pow(hBar,2)
m = 1
E = 1000
v = 0

lastX = 0
for i in range(0,int(pot.shape[0]*potStepSize/unitSize)):
    x = i*unitSize
    try:
        if pot[int(x/potStepSize)] < pot[int(x/potStepSize)+1]:
            func[i] += rand.random()*randFunc
    except:
        pass
    acc = -2*m*(pot[int(x/potStepSize)]*func[i]-E*func[i])/hBarSquare
    func = np.append(func,v*(x-lastX))
    v += acc*(x-lastX)
    lastX = x

plt.plot(func)
plt.show()
