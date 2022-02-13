## compute Theta
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# takes a 2-dim yIn
def poolData( self, yIn ): 
    # create matrix with all polynomials y^3, xy^2, x^2y, x^3, ... up to order 'polyorder' and plug in 'yIn'
    poly = PolynomialFeatures( self.polyorder )
    polyMatrix = poly.fit_transform( yIn )
    moreFunctions = [ f( yIn ) for f in self.functionDict ]

    # glue together and release into the world
    return np.concatenate( [polyMatrix,] + moreFunctions, axis=1 ) # maybe not the most beautiful/fastest solution



# same as 'poolData' but takes 1-dim arrays as input
def poolDataVec( self, yIn ):
    yIn2 = yIn.reshape(1, -1)
    
    poly = PolynomialFeatures( self.polyorder )
    polyMatrix = poly.fit_transform( yIn2 )

    if self.functionDict:
        moreFunctions = [ f( yIn2 ) for f in self.functionDict ]
        return np.concatenate( [polyMatrix,] + moreFunctions, axis=1 )
    else:
        return polyMatrix
    
# same as 'poolData' but takes 1-dim arrays as input        
def poolDataVecTry( self, yIn ):
    return poolData( self, yIn.reshape(1, -1) )
