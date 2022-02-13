## generate approx data for a pathological function with Sindy
import pandas as pd
import numpy as np

from utils.poolData import poolData
from utils.sparsifyDynamics import sparsifyDynamics
import utils.functionNorms as fn
from data.pathologicalExact import fDict
from parameters import pathological as pa

def pathologicalFunction( inName, outName ):
    ## load data
    df = pd.read_csv( f'.\\data\\pathologicalExact{pa.name}{inName}.csv', index_col=0 )
    tTrainEnd = int( pa.trainBatchSize * pa.tEnd )
    xTrain, dxTrain = df.loc[:tTrainEnd][['x']], df.loc[:tTrainEnd][['dx']]

    ## add noise to training data
    #xTrain += lo.eta * lo.randn( xTrain.shape ) 
    dxTrain += pa.eta * pa.randn( dxTrain.shape )

    ## Sindy: identify ODE
    Theta = poolData( pa, xTrain ) # pool Data (build library of nonlinear time series)
    Xi = sparsifyDynamics( pa, Theta, dxTrain.values ) # Sparse regression: sequential least squares

    """## calculate errors
    exactFunction = fDict[pa.name]
    functionDifference = lambda x: poolData( pa, x ) @ Xi - exactFunction(x)
    l2Error = fn.LpNorm( functionDifference, pa.dOmg1, pa.dOmg2, 2 )
    lInfError = fn.LInftyNorm( functionDifference, pa.dOmg1, pa.dOmg2, )"""

    ## compute 'Derivative'
    x = df[['x']]
    dx = poolData( pa, x ) @ Xi

    ## Save data
    outputData = pd.DataFrame( [], index=df.index, columns=df.columns )
    outputData[['x']], outputData[['dx']] = x, dx
    outputData.to_csv( f'.\\data\\pathologicalSindy{pa.name}{outName}.csv' )

    

