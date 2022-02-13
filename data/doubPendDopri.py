## approx solution of double pendulum with dopri5
import numpy as np
import pandas as pd
from scipy.constants import g
from parameters import doubPend as dp

def diff(t, y):
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

# if y is a vector; used in dp.solveOde
def diffVec(t, y):
        return( diff(t, y.reshape(1,-1)) )


# -------------------------------------------------------------------------------------------------
def doubPendDopri( outName ):
    t = np.arange( dp.tStart, dp.tEnd, dp.tStep )
        
    ## solve ODE
    x, rkStep = dp.solveOde( dp, diffVec, getTimeStep=True )

    ## compute Derivative from ODE+
    dx = diff(0, x)

    ## concatenate data
    # This might be a bit confusing. In order to solve the ODE numerically, we reformulated it.
    # So now new official state is [phi1, phi2, dphi1, dphi2] with derivative [dphi1, dphi2, ddphi1, ddphi2].
    # For practical reasons we only store dphi1, dphi2 in outputData once.
    outputData = pd.DataFrame( x, index=t, columns=['phi1','phi2','dphi1','dphi2'] )
    outputData['ddphi1'], outputData['ddphi2'] = dx[:,2], dx[:,3]
    outputData['rkStep'] = rkStep
    
    ## save data
    outputData.to_csv( f'.\\data\\doubPendDopri{outName}.csv' )



    
