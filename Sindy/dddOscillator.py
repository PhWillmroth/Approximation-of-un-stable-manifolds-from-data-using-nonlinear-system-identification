## generate approx data for 3D Oscillator with Sindy
import pandas as pd
from utils.poolData import poolData, poolDataVec
from utils.sparsifyDynamics import sparsifyDynamics
from parameters import dddOsc as do
import numpy as np


def dddOscillator( inName, outName ):
    ## load training data
    df = pd.read_csv( f'.\\data\\dddOscDopri{inName}.csv', index_col=0 )
    tTrainEnd = int( do.trainBatchSize * do.tEnd )
    xTrain, dxTrain = df.loc[:tTrainEnd][['x1','x2','x3']], df.loc[:tTrainEnd][['dx1','dx2','dx3']]

    ## add noise to data  
    dxTrain += do.eta * do.randn( dxTrain.shape )

    ## Sindy: identify ODE
    Theta = poolData( do, xTrain ) # pool Data (build library of nonlinear time series)
    Xi = sparsifyDynamics( do, Theta, dxTrain.values ) # compute Sparse regression: sequential least squares

    ## Solve ODE from Sindy numerically
    rhs = lambda t, x: poolDataVec( do, x ) @ Xi
    x = do.solveOde( do, rhs )

    ## compute Derivative from ODE
    dx = poolData( do, x ) @ Xi

    ## Save data
    outputData = pd.DataFrame( [], index=df.index, columns=df.columns )
    outputData[['x1','x2','x3']], outputData[['dx1','dx2','dx3']] = x, dx
    outputData.to_csv( f'.\\data\\dddOscSindy{outName}.csv' )



def getDoRank(inName, mMax, scale=1):
    ## load training data
    df = pd.read_csv( f'.\\data\\dddOscDopri{inName}.csv', index_col=0 )
    xTrain = df[['x1','x2','x3']]

    out = []
    for m in range(1, mMax):
        theta = poolData( do, xTrain.iloc[:m])
        out.append([m, np.linalg.matrix_rank(theta)])
    
    return np.array(out) / scale