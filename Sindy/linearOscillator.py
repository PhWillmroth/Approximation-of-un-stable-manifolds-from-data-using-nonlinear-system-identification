## generate approx data for linear Oscillator with Sindy
import pandas as pd
import numpy as np

from utils.poolData import poolData, poolDataVec
from utils.sparsifyDynamics import sparsifyDynamics
from utils.exactError import exactErrorSindy
from utils.functionNorms import LpNorm
from parameters import linOsc as lo

def linearOscillator( inName, outName ):
    ## load data
    #df = pd.read_csv( f'.\\data\\linOscDopri{inName}.csv', index_col=0 )
    df = pd.read_csv( f'.\\data\\linOscExact{inName}.csv', index_col=0 ) # use exact data instead
    tTrainEnd = int( lo.trainBatchSize * lo.tEnd )
    xTrain, dxTrain = df.loc[:tTrainEnd][['x1','x2']], df.loc[:tTrainEnd][['dx1','dx2']]

    ## add noise to training data
    dxTrain += lo.eta * lo.randn( dxTrain.shape )

    ## Sindy: identify ODE
    Theta = poolData( lo, xTrain ) # pool Data (build library of nonlinear time series)
    Xi = sparsifyDynamics( lo, Theta, dxTrain.values ) # Sparse regression: sequential least squares

    ## calculate errors
    """ A = np.array( [[-.1, 2], [-2, -.1]] ) # dynamics
    print("Exact Error", exactErrorSindy(lambda x, y: A @ np.array([[x,y]]).T, lo.dOmg1, lo.dOmg2, lo.polyorder))
    print("Real Error", LpNorm( lambda x: poolData(lo,x) @ Xi, lo.dOmg1, lo.dOmg2, 2 ) )
    """

    ## Solve ODE (that we obtained) from Sindy numerically
    rhs = lambda t, x: poolDataVec( lo, x ) @ Xi
    x = lo.solveOde( lo, rhs )

    ## compute Derivative from (new) ODE
    dx = poolData( lo, x ) @ Xi

    ## Save data
    outputData = pd.DataFrame( [], index=df.index, columns=df.columns )
    outputData[['x1','x2']], outputData[['dx1','dx2']] = x, dx
    outputData.to_csv( f'.\\data\\linOscSindy{outName}.csv' )


def getLoRank(inName, mMax, scale=1):
    ## load training data
    df = pd.read_csv( f'.\\data\\linOscExact{inName}.csv', index_col=0 )
    xTrain = df[['x1','x2']]

    out = []
    for m in range(1, mMax):
        theta = poolData( lo, xTrain.iloc[:m])
        out.append([m, np.linalg.matrix_rank(theta)])
    
    return np.array(out) / scale