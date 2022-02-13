## generate approx data for cubic Oscillator with Series Expansion
import numpy as np
import pandas as pd
from utils.poolDataSeries import methodDict, expandExpan2d
from utils.functionNorms import relInftyError
from parameters import cubOsc as cu

def cubOscExpan( inName, outName ):
    ## configure training method
    seriesFun = methodDict[cu.expanMethod]
    order = cu.partSumOrder
    
    ## load and cut training data
    df = pd.read_csv( f'.\\data\\cubOscDopri{inName}.csv', index_col=0 )
    tTrainEnd = int( cu.trainBatchSize * cu.tEnd )
    xTrain, dxTrain = df.loc[:tTrainEnd][['x1','x2']], df.loc[:tTrainEnd][['dx1','dx2']]

    ## add noise to data
    dxTrain += cu.eta * cu.randn( dxTrain.shape )

    ## identify ODE
    Theta = expandExpan2d( xTrain.values, order, seriesFun )
    b = np.linalg.lstsq( Theta, dxTrain.values, rcond=None )[0]

    ## Solve ODE (that we obtained) numerically
    rhs = lambda t, x: expandExpan2d(x.reshape(1, -1), order, seriesFun) @ b
    x = cu.solveOde( cu, rhs )
    
    ## compute Derivative from (new) ODE
    dx = expandExpan2d( x, order, seriesFun ) @ b

    ## Save data
    outputData = pd.DataFrame( [], index=df.index, columns=df.columns )
    outputData[['x1','x2']], outputData[['dx1','dx2']] = x, dx
    outputData.to_csv( f'.\\data\\cubOscExpan{outName}.csv' )
