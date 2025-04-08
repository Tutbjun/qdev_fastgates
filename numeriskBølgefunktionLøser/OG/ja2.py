import numpy as np
import matplotlib.pyplot as plt
import math as m

def float_range(x, y, jump):
    while x < y:
        yield x
        x += jump

def frange(x, y, jump):
    return(list(float_range(x,y,jump)))


func = np.array([])
funcCords = np.array([])
stepSize = 0.1
limits = np.array([-10,10], dtype='int32')
pots = np.array([])
for i in range(int(min(limits)/stepSize),int(max(limits)/stepSize)):
    x = i * stepSize
    if x < -1:
        pots = np.append(pots,[x,10])
    elif x >= -1 and x <= 1:
        pots = np.append(pots,[x,0])
    elif x > 1:
        pots = np.append(pots,[x,10])
    pots = np.reshape(pots, (len(pots)//2,2), order='C')
func = np.append(func,[0,1])
func = np.reshape(func, (len(func)//2,2), order='C')

hbar = 1
hbarSquare = m.pow(hbar,2)
m = 1
E = 0

for x in frange(func[0,0],max(limits),stepSize):
    i = int(x / stepSize)
    potIndex = np.where(pots == x)
    acc = -(E-pots(potIndex))*2*m/hbarSquare
    
    print(i)
    


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