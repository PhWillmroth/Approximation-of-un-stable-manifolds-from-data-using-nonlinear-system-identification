# function norms
import numpy as np
from scipy.integrate import quad, dblquad, tplquad, nquad


def pHalfNorm( x, p ):
    return (x**p).sum()

def LInftyNorm( fun, x0, x1 ):
    dim = len( x0 )
    finalMax = -1
    step = 2

    assert len(x0) == len(x1), f'LInfinity-Norm: please check input dimensions: {len(x0) != len(x1)}'

    for k in range(3):
        # create mesh
        mesh = []
        for i in range( dim ):
            mesh.append( np.linspace( x0[i], x1[i], max(5 , int((x1[i]-x0[i])/step)) ) )

        if dim == 1:
            x = mesh[0]
            x = x.reshape((x.size,1)) 
            omega = x
        elif dim == 2:
            x,y = np.meshgrid( mesh[0], mesh[1] )
            x, y = x.reshape((x.size,1)), y.reshape((y.size,1))
            omega = np.concatenate( (x,y), 1 )
        elif dim == 3:
            x,y,z = np.meshgrid( mesh[0], mesh[1], mesh[2] )
            x, y, z = x.reshape((x.size,1)), y.reshape((y.size,1)), z.reshape((z.size,1))
            omega = np.concatenate( (x,y,z), 1 )
        elif dim == 4:
            w,x,y,z = np.meshgrid( mesh[0], mesh[1], mesh[2], mesh[3] )
            w, x, y, z = w.reshape((w.size,1)), x.reshape((x.size,1)), y.reshape((y.size,1)), z.reshape((z.size,1))
            omega = np.concatenate( (w,x,y,z), 1 )
        else:
            raise ValueError( 'Die LInfinity-Norm is only implemented for dim = 1,2,3,4.' )
        
        # find max on mesh
        tempMax = np.amax( abs( fun(omega) ) )
        
        # check status
        if tempMax > finalMax:
            finalMax = tempMax
            step *= 0.5
        elif tempMax == finalMax:
            break
        return finalMax

def LpNorm( fun, x0, x1, p ):
    assert len( x0 ) == len( x1 ), 'Lp-Norm: please check the input dimensions.'

    dim = len( x0 )

    if dim == 1:
        intFun = lambda x: pHalfNorm( fun( np.array([[x,],]) ), p )
        I = quad( intFun , x0[0], x1[0] )[0]
    elif dim == 2:
        intFun = lambda x, y: pHalfNorm( fun( np.array([[y,x]]) ), p )
        I = dblquad( intFun , x0[0], x1[0], lambda x: x0[1], lambda x: x1[1] )[0]
    elif dim == 3:
        intFun = lambda x, y, z: pHalfNorm( fun( np.array([[z,y,x]]) ), p )
        I = tplquad( intFun , x0[0], x1[0], lambda x: x0[1], lambda x: x1[1],
                     lambda x,y: x0[2], lambda x,y: x1[2] )[0]
    elif dim == 4:
        intFun = lambda w, x, y, z: pHalfNorm( fun( np.array([[z,y,x,w]]) ), p )
        I = nquad( intFun, [ (x0[0],x1[0]), (x0[1],x1[1]), (x0[2],x1[2]), (x0[3],x1[3]) ] )[0]
    else:
        raise ValueError( 'The Lp-Norm is only implemented for dim = 1,2,3,4.' )
    return I ** (1./p)








def relInftyError( f1, f2, x0, x1 ):
    fun = lambda x: f1(0,x) - f2(0,x)
    return LInftyNorm( fun, x0, x1 ) / LInftyNorm( lambda x: f2(0,x), x0, x1 )

def relLpError( f1, f2, x0, x1, p):
    fun = lambda x: f1(0,x) - f2(0,x)
    return LpNorm( fun, x0, x1, p ) / LpNorm( lambda x: f2(0,x), x0, x1, p )







        
    
