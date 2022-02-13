## generate approx data for double Pendulum with Sindy
import numpy as np
import pandas as pd
from scipy.constants import g
from utils.poolData import poolData, poolDataVec
from utils.sparsifyDynamics import sparsifyDynamics
from utils.functionNorms import relInftyError, relLpError
from parameters import doubPend as dp

def doubPendDynamics(t, y):
        phi1, phi2, dphi1, dphi2 = y[:,0], y[:,1], y[:,2], y[:,3]
        
        # model parameters
        l1, l2, m1, m2 = 1, 1, 1, 1
        M = m1 + m2
   
        # calculate rhs
        C = np.cos( phi1 - phi2 )
        S = np.sin( phi1 - phi2 )

        ddphi1 = (m2*C*(l1*S*(dphi1**2)-g*np.sin(phi2)) + m2*l2*S*(dphi2**2) + \
                       M*g*np.sin(phi1)) / (m2*l1*(C**2)-M*l1)
        ddphi2 = (m2*l2*C*S*(dphi2**2) + M*l1*S*(dphi1**2) + \
                      M*g*C*np.sin(phi1) - M*g*np.sin(phi2)) / (M*l2-m2*l2*(C**2))
        return np.array([ dphi1,dphi2,ddphi1,ddphi2 ]).T



def doublePendulum( inName, outName ):
    ## load training data
    df = pd.read_csv( f'.\\data\\doubPendDopri{inName}.csv', index_col=0 )
    df = df.loc[ :int(dp.trainBatchSize * dp.tEnd) ] # cut data
    xTrain, dxTrain = df[['phi1','phi2','dphi1','dphi2']], df[['dphi1','dphi2','ddphi1','ddphi2']] # see comment in doubPendDopri

    # add noise
    dxTrain += dp.eta * dp.randn( dxTrain.shape )

    ## Sindy: identify ODE
    Theta = poolData( dp, xTrain.values )
    Xi = sparsifyDynamics( dp, Theta, dxTrain.values )
   
    '''
    ## compute rel error
    x0, x1, p = [0,0,0,0], [2*np.pi,2*np.pi,6,8], 2
    rhs = lambda t, x: poolData( dp, x ) @ Xi

    supError = relInftyError( rhs, doubPendDynamics, x0, x1 )
    LpError = -1 #relLpError( rhs, doubPendDynamics, x0, x1, p )
    proceed = input( f'Doub pend Sindy - L∞ error: {supError}, L{p} error: {LpError} \nProceed? y/n ' )

    assert proceed != 'n' # ----------------------------------------------------------------------------
    '''
    
    ## solve new ODE
    rhs = lambda t, x: poolDataVec( dp, x ) @ Xi
    x, rkStep = dp.solveOde( dp, rhs, getTimeStep=True,
                             tOde=[dp.tSindyStart,dp.tSindyEnd,dp.tSindyStep] )
    
    ## compute Derivative from ODE
    dx = poolData( dp, x ) @ Xi

    ## Save data
    outputData = pd.DataFrame( [], columns=df.columns, index=np.arange(dp.tSindyStart,dp.tSindyEnd,dp.tSindyStep) )
    outputData[['phi1','phi2']], outputData[['dphi1','dphi2','ddphi1','ddphi2']], outputData['rkStep'] = x[:,:2], dx, rkStep
    outputData.to_csv( f'.\\data\\doubPendSindy{outName}.csv' )


def getDpRank(inName, mMax, scale=1):
    ## load training data
    df = pd.read_csv( f'.\\data\\doubPendDopri{inName}.csv', index_col=0 )
    xTrain = df[['phi1','phi2','dphi1','dphi2']]

    out = []
    for m in range(1, mMax):
        theta = poolData( dp, xTrain.iloc[:m])
        out.append([m, np.linalg.matrix_rank(theta)])
    
    return np.array(out) / scale