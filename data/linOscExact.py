## generate exact data of the linear Oscillator
import numpy as np
import pandas as pd
from parameters import linOsc as lo

def linearOscillator( t ):
    x1  =  2 * np.exp( -0.1*t ) * np.cos( 2*t )
    x2  = -2 * np.exp( -0.1*t ) * np.sin( 2*t )
    return np.array([ x1, x2 ]).T

# --------------------------------------------------------------------
def linOscExact( outName ):
    # prepare stuff
    A = np.array( [[-.1, 2], [-2, -.1]] )   # dynamics
    t = np.arange( lo.tStart, lo.tEnd, lo.tStep )
    outputData = pd.DataFrame( [], index=t, columns=['x1','x2','dx1','dx2'] ) # init dataframe

    # generate data
    outputData[['x1', 'x2']] = linearOscillator( t ) # assign x

    ## compute Derivative from ODE
    outputData[['dx1', 'dx2']] = outputData[['x1', 'x2']] @ A.T

    # save data
    outputData.to_csv( f'.\\data\\linOscExact{outName}.csv' )



