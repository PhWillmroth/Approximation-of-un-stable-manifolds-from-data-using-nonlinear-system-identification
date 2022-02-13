## approx solution of cubic Oscillator with Dopri5
import numpy as np
import pandas as pd
from parameters import cubOsc as cu

def cubOscDopri( outName ):
    A = np.array([[ -0.1, 2 ],[ -2, -0.1 ]] )
    rhs = lambda t, x: A @ (x ** 3) # dynamics
    t = np.arange( cu.tStart, cu.tEnd, cu.tStep )
    outputData = pd.DataFrame( [], index=t, columns=['x1','x2','dx1','dx2'] ) # init dataframe

    ## Solve ODE
    outputData[['x1', 'x2']] = cu.solveOde( cu, rhs )

    ## compute Derivative from ODE
    outputData[['dx1', 'dx2']] = ( outputData[['x1','x2']] ** 3 ) @ A.T

    ## save approximated data
    outputData.to_csv( f'.\\data\\cubOscDopri{outName}.csv' ) 







