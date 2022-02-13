## compute partial series expansion for various types
import numpy as np
import itertools

chebPolynomials1 = [    lambda x: np.ones((x.shape[0], 1)),
                        lambda x: x,
                        lambda x: 2*(x**2) - 1,
                        lambda x: 4*(x**3) - 3*x,
                        lambda x: 8*(x**4) - 8*(x**2) +  1,
                        lambda x: 16*(x**5) - 20*(x**3) + 5*x,
                        lambda x: 32*(x**6) - 48*(x**4) + 18*(x**2) - 1,
                        lambda x: 64*(x**7) - 112*(x**5) + 56*(x**3) - 7*x,
                        lambda x: 128*(x**8) - 256*(x**6) + 160*(x**4) - 32*(x**2) + 1,
                        lambda x: 256*(x**9) - 576*(x**7) + 432*(x**5) - 120*(x**3) + 9 * x,
                        lambda x: 512*(x**10) - 1280*(x**8) + 1120*(x**6) - 400*(x**4) + 50*(x**2) - 1,
                        lambda x: 1024*(x**11) - 2816*(x**9) + 2816*(x**7) - 1232*(x**5) + 220*(x**3) - 11*x
                        ]
    
chebPolynomials2 = [    lambda x: np.ones((x.shape[0], 1)),
                        lambda x: 2 * x,
                        lambda x: 4*(x**2) - x**0,
                        lambda x: 8*(x**3) - 4*x,
                        lambda x: 16*(x**4) - 12*(x**2) +  x**0,
                        lambda x: 32*(x**5) - 32*(x**3) + 6*x,
                        lambda x: 64*(x**6) - 80*(x**4) + 24*(x**2) - x**0,
                        lambda x: 128*(x**7) - 192*(x**5) + 80*(x**3) - 8*x
                        ]
    


# -------------- Chebyshev Polynomials -----------------------

def genCheb1Iter( x, order ):
    U_n0, U_n1 = x, 1
    Theta = [ np.ones((x.shape[0], 1)) ]

    for k in range( order ):
            temp = U_n1
            U_n1 = 2 * x * U_n1 - U_n0
            U_n0 = temp
            Theta.append( U_n1 )   
    return np.concatenate( Theta, axis=1 )

def genCheb2Iter( x, order ):
    U_n0, U_n1 = 0, 1
    Theta = [ np.ones((x.shape[0], 1)) ]

    for k in range( order ):
            temp = U_n1
            U_n1 = 2 * x * U_n1 - U_n0
            U_n0 = temp
            Theta.append( U_n1 )  
    return np.concatenate( Theta, axis=1 )

def genCheb1Dict( x, order ):
    Theta = [ f(x) for f in chebPolynomials1[:(order+1)] ]
    return np.concatenate( Theta, axis=1 )

def genCheb2Dict( x, order ):
    Theta = [ f(x) for f in chebPolynomials2[:(order+1)] ]
    return np.concatenate( Theta, axis=1 )


# -------------- Fourier Series -----------------------

def genFourier( x, order ):
    N = int( order / 2 )
    si = [ np.sin( k * x ) for k in range( 1, order+1-N ) ]
    co = [ np.cos( k * x ) for k in range( N+1 ) ]
    Theta = list( itertools.chain.from_iterable( zip(si,co) ) ) # abwechselnd sin und cos
    return np.concatenate( Theta, axis=1 )



# -------------- Dirichlet Series -----------------------

def genDirichlet( x, order ):
    # return np.concatenate( [ k ** (-x) for k in range( 1, order+2 ) ], axis=1 ) #https://github.com/numpy/numpy/issues/8917
    return np.concatenate( [ 1 / (k ** x) for k in range( 1, order+2 ) ], axis=1 )



# -------------- Laurent Series -----------------------

def genLaurent( x, order ):
    N = int( order / 2 )
    #Theta = [ x**k for k in range( -N, order+1-N ) ] # s.o.
    Theta = [ 1 / (x**k) for k in range( 1, N+1 ) ] + [ x**k for k in range( order+1-N ) ]
    return np.concatenate( Theta, axis=1 )



# -------------- Power Series -----------------------

def genPower( x, order ):
    return np.concatenate( [ x**k for k in range( order+1 ) ], axis=1 )



# -------------- Legendre Series -----------------------

def genLegendre( x, order ):
    U_n0, U_n1 = 1, 1
    Theta = [ np.ones((x.shape[0], 1)) ]
    for k in range( 1, order+1 ):
            temp = U_n1
            U_n1 = (2*k-1)/k * x * U_n1 - (k-1)/k * U_n0
            U_n0 = temp
            Theta.append( U_n1 )   
    return np.concatenate( Theta, axis=1 )





# -------------- Make Series Expansion Multivariate ----
# ------------------------------------------------------
def expandExpan2d(data, mu, serializer):
    x = serializer(data[:,0,None], mu)
    y = serializer(data[:,1,None], mu)

    outList = []
    for i in range(mu+1):
        for j in range(mu-i+1):
            outList.append( x[:,i,None] * y[:,j,None] )
    return np.concatenate(outList, axis=1) 


def expandExpan3d(mu, data, serializer):
    x = serializer(data[:,0,None], mu)
    y = serializer(data[:,1,None], mu)
    z = serializer(data[:,2,None], mu)

    outList = []
    for i in range(mu+1):
        for j in range(mu-i+1):
            for k in range(mu-i-j+1):
                outList.append( x[:,i,None] * y[:,j,None] * z[:,k,None] )
    return np.concatenate(outList, axis=1)


# -------------- General ------------------------------
# -----------------------------------------------------
methodDict = {'chebyshev':genCheb1Dict, 'fourier':genFourier, 'dirichlet':genDirichlet,
                'laurent':genLaurent, 'power':genPower, 'legendre':genLegendre }