## generate approx data for double pendulum with Series Expansion
import numpy as np
import pandas as pd
from scipy.constants import g
from utils.poolDataSeries import *
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


def doubPendExpan( inName, outName ):
    methodDict = {'chebyshev':genCheb1Dict, 'fourier':genFourier, 'dirichlet':genDirichlet,
                  'laurent':genLaurent, 'power':genPower, 'legendre':genLegendre }
    genTheta = methodDict[dp.expanMethod]
    order = dp.partSumOrder
    
    ## load training data
    df = pd.read_csv( f'.\\data\\doubPendDopri{inName}.csv', index_col=0 )
    df = df.loc[ :int(dp.trainBatchSize * dp.tEnd) ]
    xTrain, dxTrain = df[['phi1','phi2','dphi1','dphi2']], df[['dphi1','dphi2','ddphi1','ddphi2']] # see comment in doubPendDopri
    
    ## add noise to data
    dxTrain += dp.eta * dp.randn( dxTrain.shape )

    ## identify ODE
    Theta = genTheta( xTrain, order )
    b = np.linalg.lstsq( Theta, dxTrain, rcond=None )[0]
    
    '''
    ## compute rel error
    x0, x1, p = [0,0,0,0], [2*np.pi,2*np.pi,6,8], 2
    rhs = lambda t, x: genTheta( x, order ) @ b
    supError = relInftyError( rhs, doubPendDynamics, x0, x1 )
    LpError = -1 #relLpError( rhs, doubPendDynamics, x0, x1, p )
    proceed = input( f'Doub Pend Expan - L∞ error: {supError}, L{p} error: {LpError} \nProceed? y/n ' )

    assert proceed != 'n' # ----------------------------------------------------------------------------
    '''
    
    ## Solve ODE (that we obtained) numerically
    rhs = lambda t, x: genTheta(x.reshape(1, -1), order) @ b
    x, rkStep = dp.solveOde( dp, rhs, getTimeStep=True,
                             tOde=[dp.tSindyStart,dp.tSindyEnd,dp.tSindyStep] )
    
    ## compute Derivative from (new) ODE
    dx = genTheta( x, order ) @ b

    ## Save data
    outputData = pd.DataFrame( [], columns=df.columns, index=np.arange(dp.tSindyStart,dp.tSindyEnd,dp.tSindyStep) ) # init df
    outputData[['phi1','phi2']], outputData[['dphi1','dphi2','ddphi1','ddphi2']], outputData['rkStep'] = x[:,:2], dx, rkStep # fill df
    outputData.to_csv( f'.\\data\\doubPendExpan{outName}.csv' ) # save df
    
