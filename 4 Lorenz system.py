#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('.\\data')
sys.path.append('.\\utils')
sys.path.append('.\\Sindy')
sys.path.append('.\\Expansion')
sys.path.append('.\\plot')

from utils.collectData import *
from utils.imgId import getImgId

# ------------------------- Helpers -----------------------------
# Display passed time
import time
def tic():
    global myTime
    myTime = time.time()

def toc( message ):
    global myTime
    timedelta = int( time.time() - myTime )
    if timedelta > 120:
        print( round(timedelta/60,1), 'min ', message )
    else:
        print( timedelta, 'sek ', message )
    myTime = time.time()


# Display all class attributes and asks 4 consent
def consent( cla ):
    attrs = vars( cla )
    print( '\n '.join( "%s: %s" % item for item in attrs.items() ) )
    input( 'Press Enter to continue...' )
    print( '\n\n' )




# ------------------------- Lorenz system -----------------------------

from data.lorenzDopri import lorenzDopri
from Sindy.lorenz import lorenz, getLzRank
from Expansion.lorenzExpansion import lorenzExpan
import plot.plotLorenz as pl
from parameters import Lorenz as lz

consent( lz )

## SETUP

# adjustable plot params
tPlotEnd_curve = [ 20, 250 ] 
etas_curve_state = [ .01, 10 ]

tPlotEndStates = 20

tPlotEndErrors = 20
etas_error = [ .0001, .001, .01, .1, 1, 10 ]
eta_names = [ '10^{-4}', '10^{-3}', '10^{-2}', '10^{-1}', '1', '10' ]
error_paths = [ f'Sindy{lz.tSindyEnd}_{e}' for e in etas_error ]

# self calculated:
lz.tEnd = max( [lz.tPlotEnd, tPlotEndStates, tPlotEndErrors] + tPlotEnd_curve )
lz.tSindyEnd = lz.tEnd
eta_list = set( etas_curve_state + etas_error)


tic()

## GENERATE DATA
# exact data
lorenzDopri( lz.tEnd ) # outName
toc( f'Lorenz dopri, tEnd={lz.tEnd}' )

# sindy data
for lz.eta in eta_list:
        lorenz( lz.tEnd, f'{lz.tSindyEnd}_{lz.eta}' ) # (inName, sindyOutName)
        toc( f'Lorenz-Sindy, eta={lz.eta}' )

# series data
for lz.eta in eta_list:
    lorenzExpan( lz.tEnd, f'{lz.tSindyEnd}_{lz.eta}' ) # (inName, expanOutName)
    toc( f'Lorenz-Expan, eta={lz.eta}' )


## PLOT
# plot curve
for lz.tPlotEnd in tPlotEnd_curve:
    # plot exact Curve
    pl.plotLorenzCurve( f'Dopri{lz.tEnd}', outName='exact' )
    toc( f'plot exact curve, t={lz.tPlotEnd}' )

    # plot sindy curve
    for lz.eta in etas_curve_state:
        pl.plotLorenzCurve( f'Sindy{lz.tSindyEnd}_{lz.eta}', outName=f'Sindy{lz.eta}' )
        toc( f'plot sindy curve, t={lz.tPlotEnd}, eta={lz.eta}' )

    # plot series curve
    for lz.eta in etas_curve_state:
        pl.plotLorenzCurve( f'Expan{lz.tSindyEnd}_{lz.eta}', outName=f'Expan{lz.eta}' )
        toc( f'plot expan curve, t={lz.tPlotEnd}, eta={lz.eta}' )


# plot states
lz.tPlotEnd = tPlotEndStates
for lz.eta in etas_curve_state:
    pl.plotLorenzState( lz.tEnd, f'{lz.tSindyEnd}_{lz.eta}', outName=lz.eta,
                        expanName=f'250_{lz.eta}' ) # exact name, sindyname
    toc( f'plot states, eta={lz.eta}')


# plot colorful errors
lz.tPlotEnd = tPlotEndErrors
pl.plotLorenzError( lz.tEnd, error_paths, eta_names, '' )
toc( f'plot colorful errors' )



# RANK
# Plot the rank of theta for the Lorenz system with different m (and fix delta t)
mMax = 2
phi = 20
out = []
stepList = [0.1, 0.01, 0.001]

for tStep in stepList:
    lz.tStep = tStep
    lz.tEnd = max(1, int(mMax*phi*tStep*1.5)) # enough buffer
    lorenzDopri( f'rk{tStep}' )
    out.append( getLzRank( f'rk{tStep}', mMax*phi, scale=phi ) )

# PLOT
imgId = getImgId( lz )
plt.vlines(1, -.01, 1.01, "grey", "--") # where we expect to hit 1

for i in range(len(stepList)):
    plt.plot(out[i][:,0], out[i][:,1], label=f"$\\Delta t = {stepList[i]}$", c=(0, i/len(stepList), 0))

plt.xlabel("$m/\\varphi$")
plt.ylabel("$\\mathrm{rk}(\\Theta(X)) / \\varphi$")
plt.legend()

plt.savefig( f'.\\plot\\rank_Lorenz_i{imgId}.pgf' )
plt.savefig( f'.\\plot\\rank_Lorenz_i{imgId}.png' )
plt.show()