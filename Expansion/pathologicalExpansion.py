## generate approx data for a pathological function with Series Expansion
import numpy as np
import pandas as pd

from utils.poolDataSeries import *
import utils.functionNorms as fn
from data.pathologicalExact import fDict
from parameters import pathological as pa


def pathologicalExpan( inName, outName ):
    # configure training method
    methodDict = {'chebyshev':genCheb1Dict, 'fourier':genFourier, 'dirichlet':genDirichlet,
                  'laurent':genLaurent, 'power':genPower, 'legendre':genLegendre }
    genTheta = methodDict[pa.expanMethod]
    order = pa.partSumOrder
    
    ## load training data
    df = pd.read_csv( f'.\\data\\pathologicalExact{pa.name}{inName}.csv', index_col=0 )
    tTrainEnd = int( pa.trainBatchSize * pa.tEnd )
    xTrain, dxTrain = df.loc[:tTrainEnd:][['x']], df.loc[:tTrainEnd][['dx']]

    ## add noise to data
    dxTrain += pa.eta * pa.randn( dxTrain.shape )

    ## identify ODE
    Theta = genTheta( xTrain, order )
    b = np.linalg.lstsq( Theta, dxTrain, rcond=None )[0]

    """## calculate errors
    functionDifference = lambda x: genTheta( x, order ) @ b - fDict[pa.name](x)
    l2Error = fn.LpNorm( functionDifference, pa.dOmg1, pa.dOmg2, 2 )
    lInfError = fn.LInftyNorm( functionDifference, pa.dOmg1, pa.dOmg2, )"""
    
    ## compute 'Derivative'
    x = df[['x']]
    dx = genTheta( x, order ) @ b

    ## Save data
    outputData = pd.DataFrame( [], index=df.index, columns=df.columns )
    outputData[['x']], outputData[['dx']] = x, dx
    outputData.to_csv( f'.\\data\\pathologicalExpan{pa.name}{outName}.csv' )

    
