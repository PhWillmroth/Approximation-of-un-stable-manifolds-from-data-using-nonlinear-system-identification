## generate approx data for Lorenz System with Sindy
import numpy as np
import pandas as pd
from utils.poolData import poolData, poolDataVec
from utils.sparsifyDynamics import sparsifyDynamics
from utils.functionNorms import relInftyError, relLpError
from parameters import Lorenz as lz

# original dynamics
def lorenzDynamics( t, state ):
    x, y, z = state[:,0], state[:,1], state[:,2]
    return np.array([ lz.sigma * (y - x),
                      x * (lz.rho - z) - y,
                      x * y - lz.beta * z ]).T


def lorenz( inName, outName ):
    ## load training data
    df = pd.read_csv( f'.\\data\\lorenzDopri{inName}.csv', index_col=0 )
    df = df.loc[ :int(lz.trainBatchSize * lz.tEnd) ] # cut data
    xTrain, dxTrain = df[['x1','x2','x3']], df[['dx1','dx2','dx3']]

    # add noise to derivative
    dxTrain += lz.eta * lz.randn( dxTrain.shape )

    ## Sindy: identify ODE
    Theta = poolData( lz, xTrain )
    Xi = sparsifyDynamics( lz, Theta, dxTrain.values )

    '''
    ## compute rel error
    x0, x1, p = [-20,-50,0], [20,50,50], 2
    rhs = lambda t, x: poolData( lz, x ) @ Xi
    supError = relInftyError( rhs, lorenzDynamics, x0, x1 )
    LpError = relLpError( rhs, lorenzDynamics, x0, x1, p )
    proceed = input( f'L∞ error: {supError} \nL{p} error: {LpError} \nProceed? y/n ' )

    assert proceed != 'n' # ----------------------------------------------------------------------------
    '''
    
    ## solve new ODE
    rhs = lambda t, x: poolDataVec( lz, x ) @ Xi
    x, rkStep = lz.solveOde( lz, rhs, getTimeStep=True,
                             tOde=[lz.tSindyStart,lz.tSindyEnd,lz.tSindyStep] )  # use different times to solve SindyOde than the ones used for training Sindy

    ## compute Derivative from (new) ODE
    dx = poolData( lz, x ) @ Xi

    ## Save data
    outputData = pd.DataFrame( [], columns=df.columns, index=np.arange( lz.tSindyStart,lz.tSindyEnd,lz.tSindyStep ) )
    outputData[['x1','x2','x3']], outputData[['dx1','dx2','dx3']], outputData['rkStep'] = x, dx, rkStep
    outputData.to_csv( f'.\\data\\lorenzExpan{outName}.csv' )  

def getLzRank(inName, mMax, scale=1):
    ## load training data
    df = pd.read_csv( f'.\\data\\lorenzDopri{inName}.csv', index_col=0 )
    xTrain = df[['x1','x2','x3']]

    out = []
    for m in range(1, mMax):
        theta = poolData( lz, xTrain.iloc[:m])
        out.append([m, np.linalg.matrix_rank(theta)])
    
    return np.array(out) / scale

