## generate approx data for linear Oscillator with Series Expansion
import numpy as np
import pandas as pd
from utils.poolDataSeries import *
from parameters import linOsc as lo


def linOscExpan( inName, outName ):
    # configure training method
    methodDict = {'chebyshev':genCheb1Dict, 'fourier':genFourier, 'dirichlet':genDirichlet,
                  'laurent':genLaurent, 'power':genPower, 'legendre':genLegendre }
    genTheta = methodDict[lo.expanMethod]
    order = lo.partSumOrder
    
    ## load training data
    df = pd.read_csv( f'.\\data\\linOscDopri{inName}.csv', index_col=0 )
    #df = pd.read_csv( f'.\\data\\cubOscExact{inName}.csv', index_col=0 ) # use exact data instead
    tTrainEnd = int( lo.trainBatchSize * lo.tEnd )
    xTrain, dxTrain = df.loc[:tTrainEnd][['x1','x2']], df.loc[:tTrainEnd][['dx1','dx2']]

    ## add noise to data
    dxTrain += lo.eta * lo.randn( dxTrain.shape )

    ## identify ODE
    Theta = genTheta( xTrain, order )
    b = np.linalg.lstsq( Theta, dxTrain, rcond=None )[0]

    ## Solve ODE (that we obtained) numerically
    rhs = lambda t, x: genTheta(x.reshape(1, -1), order) @ b
    x = lo.solveOde( lo, rhs )
    
    ## compute Derivative from (new) ODE
    dx = genTheta( x, order ) @ b

    ## Save data
    outputData = pd.DataFrame( [], index=df.index, columns=df.columns )
    outputData[['x1','x2']], outputData[['dx1','dx2']] = x, dx
    outputData.to_csv( f'.\\data\\linOscExpan{outName}.csv' )
