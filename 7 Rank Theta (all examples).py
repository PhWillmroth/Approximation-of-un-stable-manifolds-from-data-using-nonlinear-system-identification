#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('.\\data')
sys.path.append('.\\utils')
sys.path.append('.\\Sindy')
sys.path.append('.\\Expansion')
sys.path.append('.\\plot')

from matplotlib import pyplot as plt
from scipy.special import binom
from utils.imgId import getImgId

from parameters import linOsc as lo
from parameters import cubOsc as cu
from parameters import dddOsc as do
from parameters import Lorenz as lz
from parameters import doubPend as dp
from data.linOscExact import linOscExact
from data.cubOscDopri import cubOscDopri
from data.dddOscDopri import dddOscDopri
from data.lorenzDopri import lorenzDopri
from data.doubPendDopri import doubPendDopri
from Sindy.linearOscillator import getLoRank
from Sindy.cubicOscillator import getCuRank
from Sindy.dddOscillator import getDoRank
from Sindy.lorenz import getLzRank
from Sindy.doublePendulum import getDpRank



# Plot the rank of theta for different m (and fix delta t)


mMax = 13
tStep = 0.001
tEnd = 100

lo.tStep, lo.tEnd = tStep, tEnd
linOscExact( 'rk' )
phi1 = int(binom(lo.n(lo) + lo.polyorder, lo.polyorder))
loRk = getLoRank('rk', mMax*phi1, scale=phi1 )

cu.tStep, cu.tEnd = tStep, tEnd
cubOscDopri( 'rk' )
phi2 = int(binom(cu.n(cu) + cu.polyorder, cu.polyorder))
cuRk = getCuRank( 'rk', mMax*phi2, scale=phi2 )

do.tStep, do.tEnd = tStep, tEnd
dddOscDopri( 'rk' )
phi3 = int(binom(do.n(do) + do.polyorder, do.polyorder))
doRk = getDoRank( 'rk', mMax*phi3, scale=phi3 )

lz.tStep, lz.tEnd = tStep, tEnd
lorenzDopri( 'rk' )
phi4 = int(binom(lz.n(lz) + lz.polyorder, lz.polyorder))
lzRk = getLzRank( 'rk', mMax*phi4, scale=phi4 ) 

dp.tStep, dp.tEnd = tStep, tEnd
doubPendDopri( 'rk' )
phi5 = int(binom(dp.n(dp) + dp.polyorder, dp.polyorder))
dpRk = getDpRank( 'rk' , mMax*phi5, scale=phi5 )

# PLOT
imgId = getImgId( lo )
plt.vlines(1, -.01, 1.01, "grey", "--") # where we expect to hit 1

plt.plot(loRk[:,0], loRk[:,1], label="Linear oscillator")
plt.plot(cuRk[:,0], cuRk[:,1], label="Cubic oscillator")
plt.plot(doRk[:,0], doRk[:,1], label="3D linear oscillator")
plt.plot(lzRk[:,0], lzRk[:,1], label="Lorenz system")
plt.plot(dpRk[:,0], dpRk[:,1], label="Double pendulum")

plt.xlabel("$m/\\varphi$")
plt.ylabel("$\\mathrm{rk}(\\Theta(X)) / \\varphi$")
plt.legend()

plt.savefig( f'.\\plot\\rank_i{imgId}.pgf' )
plt.savefig( f'.\\plot\\rank_i{imgId}.png' )
plt.show()