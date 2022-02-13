## generate approx data for lorenz system with series expansion
import numpy as np
import pandas as pd
from utils.functionNorms import relInftyError, relLpError
from utils.poolDataSeries import *
from parameters import Lorenz as lz

# original dynamics
def lorenzDynamics( t, state ):
    x, y, z = state[:,0], state[:,1], state[:,2]
    return np.array([ lz.sigma * (y - x),
                      x * (lz.rho - z) - y,
                      x * y - lz.beta * z ]).T


def lorenzExpan( inName, outName ):
    ## configure training method
    methodDict = {'chebyshev':genCheb1Dict, 'fourier':genFourier, 'dirichlet':genDirichlet,
                  'laurent':genLaurent, 'power':genPower, 'legendre':genLegendre }
    genTheta = methodDict[lz.expanMethod]
    order = lz.partSumOrder
    
    ## load and cut training data
    df = pd.read_csv( f'.\\data\\lorenzDopri{inName}.csv', index_col=0 )
    df = df.loc[ :int(lz.trainBatchSize * lz.tEnd) ]
    xTrain, dxTrain = df[['x1','x2','x3']], df[['dx1','dx2','dx3']]

    ## add noise to data
    dxTrain += lz.eta * lz.randn( dxTrain.shape )

    ## identify ODE
    Theta = genTheta( xTrain, order )
    b = np.linalg.lstsq( Theta, dxTrain, rcond=None )[0]

    '''
    ## compute rel error
    x0, x1, p = [-20,-50,0], [20,50,50], 2
    rhs = lambda t, x: genTheta( x, order ) @ b
    supError = relInftyError( rhs, lorenzDynamics, x0, x1 )
    LpError = -1 #relLpError( rhs, lorenzDynamics, x0, x1, p ) #slow
    proceed = input( f'L∞ error: {supError} \nL{p} error: {LpError} \nProceed? y/n ' )
    
    assert proceed != 'n' # ----------------------------------------------------------------------------
    '''
    
    ## Solve ODE (that we obtained) numerically
    rhs = lambda t, x: genTheta(x.reshape(1, -1), order) @ b
    x, rkStep = lz.solveOde( lz, rhs, getTimeStep=True,
                             tOde=[lz.tSindyStart,lz.tSindyEnd,lz.tSindyStep] )  # use different times to solve SindyOde than the ones used for training Sindy

    
    ## compute Derivative from (new) ODE
    dx = genTheta( x, order ) @ b

    ## Save data
    outputData = pd.DataFrame( [], columns=df.columns, index=np.arange( lz.tSindyStart,lz.tSindyEnd,lz.tSindyStep ) )
    outputData[['x1','x2','x3']], outputData[['dx1','dx2','dx3']], outputData['rkStep'] = x, dx, rkStep
    outputData.to_csv( f'.\\data\\lorenzExpan{outName}.csv' )
