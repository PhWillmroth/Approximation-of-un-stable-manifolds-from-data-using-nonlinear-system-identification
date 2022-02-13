## approx solution of linear Oscillator with dopri5
import numpy as np
import pandas as pd
from parameters import linOsc as lo

def linOscDopri( outName ):
    A = np.array( [[-.1, 2], [-2, -.1]] )   # dynamics
    rhs = lambda t, x: A @ x
    t = np.arange( lo.tStart, lo.tEnd, lo.tStep )
    outputData = pd.DataFrame( [], index=t, columns=['x1','x2','dx1','dx2'] ) # init dataframe
    
    ## Solve ODE
    outputData[['x1', 'x2']] = lo.solveOde( lo, rhs )

    ## compute Derivative from ODE
    outputData[['dx1', 'dx2']] = outputData[['x1','x2']] @ A.T

    ## save approximated data
    outputData.to_csv( f'.\\data\\linOscDopri{outName}.csv' )









