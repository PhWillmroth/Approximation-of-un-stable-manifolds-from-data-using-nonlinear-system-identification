import pandas as pd
from utils.tossiReduceDimension import reduceDimension
from utils.poolData import poolData
from utils.sparsifyDynamics import sparsifyDynamics
from utils.NoiseReductionDerivativeCalculation import TVRegDiff
from parameters import spiral as sp
from matplotlib import pyplot as plt


def spiralTossi( inName, outName ):

    ## load spiral data
    data = pd.read_csv( f'.\\data\\spiralExact{inName}.csv', index_col=0 )
    data = data[['x1','x2','y']]
    #dataDerivative = TVRegDiff( data[['x1','x2']].values.flatten(), 10, 10, scale='large', plotflag=False ) # compute derivative from data

    ## add noise
    data = data + sp.eta * sp.randn( data.shape )
    
    ## reduce dimension (of the whole input, including the data of which I want to predict the output)
    t0, t1 = sp.tTossiStart, sp.tTossiEnd
    transfData = reduceDimension( sp, data[[ 'x1','x2' ]] )

    # plot
    '''
    #plt.plot(data['x1'], label='x1')
    #plt.plot(data['x2'], label='x2')
    plt.plot(data['y'], label='y')
    plt.plot(transfData['z'], label="z")
    plt.ylim( np.amin(transfData['z'].values), np.amax(transfData['z'].values) )
    plt.legend()
    plt.title( sp.method )
    plt.show()
    '''
    
    ## find mapping from intrinsic coords to output y (i.e. we 'train' our model here)
    trainInput = transfData[t0:t1]
    desiredOutput = data.loc[t0:t1, ['y']].to_numpy()
    
    # regression
    regressor = poolData( sp, trainInput )
    b = sparsifyDynamics( sp, regressor, desiredOutput )

    # f maps z (and not x) to yApprox
    f = lambda z: poolData(sp, z) @ b 

    ## approximate new data with Tossi model
    yApprox = f( transfData )
    approxData = data.copy() 
    approxData['y'] = yApprox

    # save data
    approxData.to_csv( f'.\\data\\spiralTossi{outName}.csv'  )
