## generate approx data for 3d Oscillator with Series Expansion
import numpy as np
import pandas as pd
from utils.poolDataSeries import *
from parameters import dddOsc as do


def dddOscExpan( inName, outName ):
    ## configure training method
    methodDict = {'chebyshev':genCheb1Dict, 'fourier':genFourier, 'dirichlet':genDirichlet,
                  'laurent':genLaurent, 'power':genPower, 'legendre':genLegendre }
    genTheta = methodDict[do.expanMethod]
    order = do.partSumOrder
    
    ## load training data
    df = pd.read_csv( f'.\\data\\dddOscDopri{inName}.csv', index_col=0 )
    tTrainEnd = int( do.trainBatchSize * do.tEnd )
    xTrain, dxTrain = df.loc[:tTrainEnd][['x1','x2','x3']], df.loc[:tTrainEnd][['dx1','dx2','dx3']]

    ## add noise to data
    dxTrain += do.eta * do.randn( dxTrain.shape )

    ## identify ODE
    Theta = genTheta( xTrain, order )
    b = np.linalg.lstsq( Theta, dxTrain, rcond=None )[0]

    ## Solve ODE (that we obtained) numerically
    rhs = lambda t, x: genTheta(x.reshape(1, -1), order) @ b
    x = do.solveOde( do, rhs )
    
    ## compute Derivative from (new) ODE
    dx = genTheta( x, order ) @ b

    ## Save data
    outputData = pd.DataFrame( [], index=df.index, columns=df.columns )
    outputData[['x1','x2','x3']], outputData[['dx1','dx2','dx3']] = x, dx
    outputData.to_csv( f'.\\data\\dddOscExpan{outName}.csv' )
