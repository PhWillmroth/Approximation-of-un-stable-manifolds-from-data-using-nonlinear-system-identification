## approx solution of Lorenz system with dopri5
import numpy as np
import pandas as pd
from parameters import Lorenz as lz

def lorenz( t, state ):
    x, y, z = state
    return np.array([ lz.sigma * (y - x),
                      x * (lz.rho - z) - y,
                      x * y - lz.beta * z ])

# -------------------------------------------------------------------------------------------------
def lorenzDopri( outName ):
    ## init
    t = np.arange( lz.tStart, lz.tEnd, lz.tStep )
    outputData = pd.DataFrame( [], index=t, columns=['x1','x2','x3','dx1','dx2','dx3','rkStep'] )

    ## solve ODE
    x, rkStep = lz.solveOde( lz, lorenz, getTimeStep=True )

    outputData[['x1','x2','x3']] = x
    outputData[['dx1','dx2','dx3']] = lorenz( 0, x.T ).T # compute Derivative from ODE
    outputData['rkStep'] = rkStep

    ## save data
    outputData.to_csv( f'.\\data\\lorenzDopri{outName}.csv' )
