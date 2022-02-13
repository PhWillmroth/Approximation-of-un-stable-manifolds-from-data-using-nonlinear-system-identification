## approx solution of 3d linear Oscillator with dopri5
import numpy as np
import pandas as pd
from parameters import dddOsc as do

def dddOscDopri( outName ):
    # Rem.: In the paper by Brunton et al. they state to use A, but use A.T in the algorithm instead.
    # Thus also the figure in the paper is mirrored.
    A  = np.array( [[-.1, -2, 0], [2, -.1, 0], [0, 0, -.3]] ) # dynamics
    rhs = lambda t, x: A @ x
    t = np.arange( do.tStart, do.tEnd, do.tStep )
    outputData = pd.DataFrame( [], index=t, columns=['x1','x2','x3','dx1','dx2','dx3'] ) # init dataframe

    ## Solve ODE
    outputData[['x1', 'x2', 'x3']] = do.solveOde( do, rhs )

    ## compute Derivative from ODE
    outputData[['dx1', 'dx2', 'dx3']] = outputData[['x1', 'x2', 'x3']] @ A.T
    
    ## save approximated data
    outputData.to_csv( f'.\\data\\dddOscDopri{outName}.csv' )







