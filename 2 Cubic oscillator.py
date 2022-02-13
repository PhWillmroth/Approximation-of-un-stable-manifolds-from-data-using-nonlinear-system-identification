#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('.\\data')
sys.path.append('.\\utils')
sys.path.append('.\\Sindy')
sys.path.append('.\\Expansion')
sys.path.append('.\\plot')

from utils.collectData import *
from utils.paduaPoints import generatePP


#2D Cubic Oscillator
from data.cubOscDopri import cubOscDopri
from Sindy.cubicOscillator import cubicOscillator, cubicOscillatorMultiple, cubicOscillatorMultiple2, getCuRank
from Expansion.cubOscExpansion import cubOscExpan
from plot.plotCubicOscillator import *
from parameters import cubOsc as cu

cubOscDopri( '' )
cubicOscillator( '', '' )
cubOscExpan( '', '' )
plotCubicOscillator( '', '', show=True )
plotCubicOscillator( '', '', includeExpan=True, show=True )
plotCuDerivative('', '', show=True )
plotCuFindings( 'cubOscv1', 'Sindy', t='eta', ylim=None, norm='L2', outName='', show=True, startInd=60 )
plotCuFindings( 'cubOscv2', 'Sindy', t='lmbd', ylim=None, norm='L2', outName='', show=True )


# Legendre fct. dict.
cu.functionDict = [
    lambda x: (0.5 * (3 * (x[:,0]**2) - 1))[None].T,
    lambda x: (0.5 * (3 * (x[:,1]**2) - 1))[None].T,
    lambda x: (x[:,0] * x[:,1])[None].T,
    lambda x: (x[:,0] * 0.5 * (3 * (x[:,1]**2) - 1))[None].T,
    lambda x: (x[:,1] * 0.5 * (3 * (x[:,0]**2) - 1))[None].T,
    lambda x: (0.5 * ( 5 * (x[:,0] ** 3) - (3 * x[:,0]) ))[None].T,
    lambda x: (0.5 * ( 5 * (x[:,1] ** 3) - (3 * x[:,1]) ))[None].T]


#####################################
# FIGURE 3.13
#####################################
cycles = 2500    
cu.polyorder = 1
cu.trainBatchSize = 1
cu.tEnd = 25

# Legendre
cu.functionDict = [
    lambda x: (0.5 * (3 * (x[:,0]**2) - 1))[None].T,
    lambda x: (0.5 * (3 * (x[:,1]**2) - 1))[None].T,
    lambda x: (x[:,0] * x[:,1])[None].T,
    lambda x: (x[:,0] * 0.5 * (3 * (x[:,1]**2) - 1))[None].T,
    lambda x: (x[:,1] * 0.5 * (3 * (x[:,0]**2) - 1))[None].T,
    lambda x: (0.5 * ( 5 * (x[:,0] ** 3) - (3 * x[:,0]) ))[None].T,
    lambda x: (0.5 * ( 5 * (x[:,1] ** 3) - (3 * x[:,1]) ))[None].T]


for Kmax in [10, 60]:
    version = f'G2-{Kmax}'

    clearOldFindings(cu, version)
    divisorsOfKmax = [i for i in range(1, Kmax+1) if Kmax % i == 0]

    for K in divisorsOfKmax:
        m = Kmax / K
        out = cubicOscillatorMultiple2( "burst", K, cycles, m )

        saveFindings( cu, version, Kmax=Kmax, K=K, init='random-1+1', polyorder=3, m=m, cycles=cycles, L2Error=out )




df1 = pd.read_pickle( f'data\\errordata_cubOscvG2-10.pkl' )
df3 = pd.read_pickle( f'data\\errordata_cubOscvG2-60.pkl' )

plt.plot(df3["K"], df3["L2Error"] / 418117.35926181235, label="Number of samples $=60$", c="green") #plt.plot(df1["K"], df1["L2Error"] / 418117.35926181235, label="Number of samples $=10$", c="green")


plt.yscale('log')
plt.legend()
plt.xlabel('Number of bursts $K$')
plt.ylabel('rel. $L^2$-error')

plt.savefig( f'.\\plot\\errordata_cubOscvG2-60.png' ) #plt.savefig( f'.\\plot\\errordata_cubOscvG2-10.png' )
plt.savefig( f'.\\plot\\errordata_cubOscvG2-60.pgf' ) #plt.savefig( f'.\\plot\\errordata_cubOscvG2-10.pgf' )
plt.show()









#####################################
# FIGURE 3.12
#####################################
cycles = 1000
version = 'H21'

cu.polyorder = 4
cu.trainBatchSize = 1
cu.tEnd = 25
cu.tStep = .1

pps = generatePP(cu.polyorder) # padua points
K = len(pps)




# for rel L2 error
from utils.functionNorms import LpNorm
A =  np.array([[-0.1,2],[-2,-0.1]])
fun = lambda x: (x ** 3) @ A.T
norm = LpNorm(fun, cu.dOmg1, cu.dOmg2, 2)


# PADUA
clearOldFindings(cu, version+"-1")

# generate "observed" samples
k = 1
for pp in pps:
    cu.x0 = pp
    cubOscDopri( f'burst{k}PPno2' )
    k += 1

for m in range(25):
    print("############## m = ", m)

    # sindy (returns L2 error)
    out = cubicOscillatorMultiple('burst', 1, K, cycles, m, postfix="PPno2")
    print("L2", out, "rel. L2", out/norm)

    saveFindings( cu, version + "-1", K=K, m=m, init='padua', cycles=cycles, L2Error=out )


# RANDOM
# generate "observed" samples
clearOldFindings(cu, version+"-2")

for k in range(1, K+1):
    cu.x0 = np.random.rand(2) * 2 - np.array([1,1])
    cubOscDopri( f'burst{k}no2' )

for m in range(25):
    print("############## m = ", m)

    # sindy (returns L2 error)
    out = cubicOscillatorMultiple('burst', 1, K, cycles, m, postfix="no2")
    print("L2", out, "rel. L2", out/norm)

    saveFindings( cu, version + "-2", K=K, m=m, init='random-1+1', cycles=cycles, L2Error=out ) 

plotMultipleFindings4(f'cubOscv{version}-1', "cubOscv{version}-2", norm)









version = 'D4'
cycles = 1000

cu.polyorder = 1
cu.trainBatchSize = 1
cu.tEnd = 25
cu.dOmg1, cu.dOmg2 = [0,]*2, [25,]*2
mu = 3

def lc(t):
    return round( np.cos(mu * t), 10) , round( np.cos((mu+1) * t), 10 )
pp1 = {  lc(np.pi * i / mu / (mu+1) ) for i in range(mu * (mu+1) + 1) }
pp2 = [] # padua points
for i in pp1:
    pp2.append( np.array(i) )

clearOldFindings(cu, version)

k = 1
for ppoint in pp2:
    cu.x0 = ppoint
    cubOscDopri( f'burst{k}PP' )
    k += 1
assert np.arange(cu.tStart, cu.tEnd, cu.tStep).size > 100

for m in range(1,100,2):
    out = cubicOscillatorMultiple('burst', 1, 10, cycles, m)
    saveFindings( cu, version, K=10, m=m, init='padua', cycles=cycles, L2Error=out )
    




from utils.functionNorms import LpNorm
A =  np.array([[-0.1,2],[-2,-0.1]])
fun = lambda x: (x ** 3) @ A.T
norm = LpNorm(fun, cu.dOmg1, cu.dOmg2, 2)
#plotMultipleFindings2(f'cubOscv{version}', 10, norm)
plotMultipleFindings3(f'cubOscv{version}', norm)




# Plot the position of the initial conditions
mMax = 45
plotInitialisation("", [f"burst{i}no2" for i in range(1, 16)], mMax, outName="random", plotNormal=False)
plotInitialisation("", [], mMax, outName="normal", )