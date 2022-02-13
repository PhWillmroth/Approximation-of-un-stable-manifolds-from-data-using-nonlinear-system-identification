## helper functions for L2/l2 calculations
import numpy as np
from scipy.integrate import quad, dblquad, tplquad, nquad
from matplotlib import pyplot as plt

def L2Product( fun1, fun2, x0, x1 ):
    assert len ( x0 ) == len( x1 ), 'L2-Produkt: Da stimmt was mit den Inputdimensionen nicht.'
    dim = len( x0 )

    if dim == 1:
        intFun = lambda x: fun1( x ) * fun2( x )
        return quad( intFun, x0[0], x1[0] )[0]
    elif dim == 2:
        intFun = lambda y, x: fun1( x, y ).T @ fun2( x, y )
        return dblquad( intFun, x0[0], x1[0], lambda x: x0[1], lambda x: x1[1] )[0]
    elif dim == 3:
        intFun = lambda z, y, x: fun1( x, y, z ).T @ fun2( x, y, z )
        return tplquad( intFun, x0[0], x1[0], lambda x: x0[1], lambda x: x1[1],
                     lambda x,y: x0[2], lambda x,y: x1[2] )[0]
    elif dim == 4:
        intFun = lambda z, y, x, w: fun1( w, x, y, z ).T @ fun2( w, x, y, z )
        return nquad( intFun, [ (x0[0],x1[0]), (x0[1],x1[1]), (x0[2],x1[2]), (x0[3],x1[3]) ] )[0]

    raise ValueError( f"Dim = {dim} is not implemented.")





def l2Error( path1, path2, **kwargs ):

    # load data
    vec1 = np.genfromtxt('.\\data\\{}.csv'.format(path1), delimiter=',')[:,:2]
    vec2 = np.genfromtxt('.\\data\\{}.csv'.format(path2), delimiter=',')[:,:2]

    # compute error
    differ = vec1 - vec2
    differNorm = [ np.linalg.norm( line ) for line in differ ]

    print( 'error = {}'.format(np.sum(differNorm)) )

    # plot
    fig = plt.figure()
    plt.yscale( 'log' ) # use logarithmic scale
    plt.title( 'Error with logarithmic scale' )
    plt.plot( differNorm )
    plt.show()

    if kwargs.get( 'returnVec', False ):
        return differNorm
