## generate approx data for cubic Oscillator with Sindy
#from utils.functionNorms import LInftyNorm
import pandas as pd
from utils.poolData import poolData, poolDataVec
from utils.sparsifyDynamics import sparsifyDynamics
from parameters import cubOsc as cu
from utils.functionNorms import LpNorm
import numpy as np

"""ALegendre = np.array([[0.,         0.        ],
    [0.,         -1.2       ],
    [1.2,        0.        ],
    [0.,         0.        ],
    [0.,         0.        ],
    [0.,         0.        ],
    [0.,         0.        ],
    [0.,         0.        ],
    [0.,         -0.8       ],
    [0.8,        0.        ]])"""

def cubicOscillator( inName, outName ):
    ## load training data
    df = pd.read_csv( f'.\\data\\cubOscDopri{inName}.csv', index_col=0 )
    tTrainEnd = int( cu.trainBatchSize * cu.tEnd )
    xTrain, dxTrain = df.loc[:tTrainEnd][['x1','x2']], df.loc[:tTrainEnd][['dx1','dx2']]

    ## add noise to data
    dxTrain += cu.eta * cu.randn( dxTrain.shape )

    ## Sindy: identify ODE
    Theta = poolData( cu, xTrain ) # pool Data (i.e., build library of nonlinear time series)
    Xi = sparsifyDynamics( cu, Theta, dxTrain.values ) # compute Sparse regression: sequential least squares

    # calculate error
    #A =  np.array([[-0.1,2],[-2,-0.1]]) # dynamics
    #diff = lambda x: ( poolData( cu, x ) @ Xi ) - ( (x ** 3) @ A.T )
    #return LpNorm(diff, cu.dOmg1, cu.dOmg2, 2)
    
    ## Solve the ODE (that we obtained from Sindy) numerically
    rhs = lambda t, x: poolDataVec( cu, x ) @ Xi
    x = cu.solveOde( cu, rhs )

    ## compute Derivative from (new) ODE
    dx = poolData( cu, x ) @ Xi

    ## Save data
    outputData = pd.DataFrame( [], index=df.index, columns=df.columns )
    outputData[['x1','x2']], outputData[['dx1','dx2']] = x, dx
    outputData.to_csv( f'.\\data\\cubOscSindy{outName}.csv' )


# For [Schaeffer et al. 2018]
def cubicOscillatorMultiple( inName, KMin, KMax, repeat, m, postfix="" ):
    xTrain, dxTrain = pd.DataFrame(), pd.DataFrame()
    m = int(m)

    ## load and stack training data
    for K in range(KMin, KMax+1):
        # load training data
        df = pd.read_csv( f'.\\data\\cubOscDopri{inName}{K}{postfix}.csv', index_col=0 ) #df = pd.read_csv( f'.\\data\\cubOscDopri{inName}{K}PP.csv', index_col=0 )
        if df.index.size < m+1: print("Oh no")
        xTrainTemp, dxTrainTemp = df.iloc[:m+1][['x1','x2']], df.iloc[:m+1][['dx1','dx2']]

        # stack input data together
        xTrain = pd.concat((xTrain, xTrainTemp))
        dxTrain = pd.concat((dxTrain, dxTrainTemp))

    ## SINDyfy
    Theta = poolData( cu, xTrain.values ) # pool Data (i.e., build library of nonlinear time series)
    out = 0
    A =  np.array([[-0.1,2],[-2,-0.1]]) # dynamics
    for r in range(repeat):
        # add noise to data
        dxTrain2 = dxTrain + cu.eta * cu.randn( dxTrain.shape ) 
        Xi = sparsifyDynamics( cu, Theta, dxTrain2.values ) # compute Sparse regression: sequential least squares

        # calculate error
        diff = lambda x: ( poolData( cu, x ) @ Xi ) - ( (x ** 3) @ A.T )
        out += LpNorm(diff, cu.dOmg1, cu.dOmg2, 2)
    
    return out / repeat
    
# For [Schaeffer et al. 2018]
def cubicOscillatorMultiple2( inName, K, cycles, m ):
    xTrain, dxTrain = pd.DataFrame(), pd.DataFrame()
    m = int(m)

    ## load and stack training data
    for iterK in range(1, K+1):
        # load training data
        df = pd.read_csv( f'.\\data\\cubOscDopri{inName}{iterK}.csv', index_col=0 )
        assert df.index.size >= m
        xTrainTemp, dxTrainTemp = df.iloc[:m+1][['x1','x2']], df.iloc[:m+1][['dx1','dx2']]

        # stack input data together
        xTrain = pd.concat((xTrain, xTrainTemp))
        dxTrain = pd.concat((dxTrain, dxTrainTemp))

    ## SINDyfy
    Theta = poolData( cu, xTrain.values ) # pool Data (i.e., build library of nonlinear time series)
    out = 0
   
    A =  np.array([[-0.1,2],[-2,-0.1]]) # dynamics
    for r in range(cycles):
        # add noise to data
        dxTrain2 = dxTrain + cu.eta * cu.randn( dxTrain.shape ) 

        ## Sindy: identify ODE
        Xi = sparsifyDynamics( cu, Theta, dxTrain2.values ) # compute Sparse regression: sequential least squares

        # calculate error
        diff = lambda x: ( poolData( cu, x ) @ Xi ) - ( (x ** 3) @ A.T )
        out += LpNorm(diff, cu.dOmg1, cu.dOmg2, 2)
    
    return out / cycles
    

def getCuRank(inName, mMax, scale=1):
    ## load training data
    df = pd.read_csv( f'.\\data\\cubOscDopri{inName}.csv', index_col=0 )
    xTrain = df[['x1','x2']]

    out = []
    for m in range(1, mMax):
        theta = poolData( cu, xTrain.iloc[:m])
        out.append([m, np.linalg.matrix_rank(theta)])
    
    return np.array(out) / scale