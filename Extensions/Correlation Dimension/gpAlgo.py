import numpy as np
from matplotlib import pyplot as plt

# Implementation of the Grassberger-Procaccia algorithm, a common 
# estimator for the correlation dimension of some set (or time series) X
# http://www.scholarpedia.org/article/Grassberger-Procaccia_algorithm

# helper fct
def Cfun( X, r ):
    leng = len( X )

    chromatic = np.array([ X[i] for j in range(leng) for i in range(j) ])
    grouped = np.array([ X[k] for k in range(leng) for j in range(k)] )
    diff = np.linalg.norm(chromatic - grouped, axis=1)[None].T

    return 2 / leng / (leng - 1) * np.count_nonzero(diff < r, axis=0)

# finds rMax itself
def gpAlgoTuned(X, numberOfR=100, plot=False):
    # find max distance between 2 consecutive samples
    maxDist = 0
    for i in range(len(X) - 1):
        tempDist = np.linalg.norm(X[i] - X[i+1])
        if maxDist < tempDist: maxDist = tempDist
    maxDist = max(1, int(maxDist) + 1)

    return gpAlgo(X, np.exp(-12), maxDist, numberOfR, plot)



# MAIN
# Takes a list of arrays [(x1, x2, x3, x4), (x1, x2, x3, x4), ...] or 2D array [[], [], [], ...]
def gpAlgo(X, rMin, rMax, numberOfR=100, plot=False):
    r = np.exp(np.linspace(np.log(rMin), np.log(rMax), numberOfR)) # should be equally spaced in log-plot
    result = Cfun(X, r)

    # omit zero entries (for the log)
    inds = ~((r == 0) + (result == 0))
    r, result = r[inds], result[inds]

    c, d = np.polyfit( np.log(r), np.log(result), 1 )[:2]

    if plot:
        plt.plot(r, result, linestyle=" ", marker="+", c="green")
        plt.plot(r, np.exp(c * np.log(r) + d), c="red", label="Fitted line") # fitted line

        plt.xlabel('$r$'), plt.ylabel('$\hat C(r)$')
        plt.xscale('log'), plt.yscale('log')
        
        plt.legend()
        plt.show()
   
    return c


# MVP
""" SIZE = 1000

One = np.eye(4)
X = [np.random.rand() * One[0] + np.random.rand() * One[1] + np.random.rand() * One[2] + np.random.rand() * One[3] for i in range(SIZE)]
c = gpAlgo(X, 0, 1, 1000)
print(c) """

