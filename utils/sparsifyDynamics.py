## compute Sparse regression with sequential (thresholded) least squares
from numpy.linalg import lstsq

def sparsifyDynamics( self, Theta, dXdt ):
    lmbd = self.lmbd
    n = dXdt.shape[1]
    
    Xi = lstsq( Theta, dXdt, rcond=None )[0] # inital guess

    for k in range( 10 ):
        smallinds = ( abs(Xi) < lmbd )
        Xi[ smallinds ] = 0
        
        for ind in range( n ):
            biginds = ~smallinds[ :, ind ]

            # Regress dynamics onto remaining terms to find sparse Xi
            Xi[biginds, ind] = lstsq( Theta[:,biginds], dXdt[:,ind], rcond=None )[0]
            
    return Xi




# Does the same as 'sparsifyDynamics' but returns the number of iterations, after which it converged.
# I.e. k=1 corresponds with least squares and then 1 repetition with the indices bigger than lambda
def countIterations( self, Theta, dXdt ):
    lmbd = self.lmbd
    n = dXdt.shape[1]
    tempinds = -1
    
    Xi = lstsq( Theta, dXdt, rcond=None )[0]

    for k in range( 10 ):
        smallinds = ( abs(Xi) < lmbd )
        Xi[ smallinds ] = 0

        if (tempinds == smallinds).all():
            return k
        else:
            tempinds = smallinds
        
        for ind in range( n ):
            biginds = ~smallinds[ :, ind ]
            Xi[biginds, ind] = lstsq( Theta[:,biginds], dXdt[:,ind], rcond=None )[0]
            
    return k


