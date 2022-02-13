import numpy as np
from utils.L2Functions import L2Product


# Calculate the value of the analytically determined error for sindy (for the case of a polynomial fct dict only).
# Here 'fun' is the rhs of the ode, x0=[a1, ..., an] and x1=[b1, ..., bn] are the integrations limits (dOmega).
def exactErrorSindy( clazz, fun, order ):
    dim = len( clazz.dOmg1 )
    x0 = clazz.dOmg1
    x1 = clazz.dOmg2
    
    ## set up Gramian G(v1, ..., vn)        
    indices = [ i for i in range( order + 1 ) ]
    ijMesh =  np.meshgrid( indices, indices)
    exponent = ijMesh[0] + ijMesh[1] + np.ones( (order+1,) * 2 )
    gramianV = np.ones( (order+1,) * 2 )
    for n in range( dim ):
        gramianV *= ( (x1[n] ** exponent) - (x0[n] ** exponent) ) / exponent

    ## set up vector [(g,g), (g, v1), ..., (g,vn)]    
    vectorWithG = np.zeros( order + 2 )
    vectorWithG[0] = L2Product( fun, fun, x0, x1 )
    for i in range( order + 1 ):
        vectorWithG[i+1] = L2Product( fun, lambda *kwargs, i=i: np.array([kwargs]).T**i, x0, x1 )

    ## set up Gramian G(g, v1, ..., vn), glue everything together
    addedLeft = np.concatenate( (vectorWithG.reshape(-1,1)[1:], gramianV), axis=1 )
    gramianVWithG = np.concatenate(( vectorWithG.reshape(1,-1), addedLeft ))

    ## calculate detG(g, v1, ..., vn) and return
    detGramianV = np.linalg.det( gramianV )
    detGramianVWithG = np.linalg.det( gramianVWithG )
  
    if ( detGramianV <= 0 or detGramianVWithG < 0): print( "WARNING: detGramianV or detGramianVWithG are <= 0" )
    return np.inf if detGramianV == 0 else ( np.sqrt( np.abs(detGramianVWithG) ) / np.sqrt( np.abs(detGramianV) ) )




def exactErrorExpan( clazz, fun ):
    expanMethod = clazz.expanMethod
    order = clazz.partSumOrder
    x0 = clazz.dOmg1
    x1 = clazz.dOmg2
    N = int( order / 2 )

    if expanMethod in ['chebyshev', 'power', 'legendre']:
        return exactErrorSindy( clazz, fun, order ) # faster

    #https://stackoverflow.com/questions/452610/how-do-i-create-a-list-of-python-lambdas-in-a-list-comprehension-for-loop
    if expanMethod == 'fourier':
        basis =  [lambda *kwargs, k=k: np.sin(k*np.array([kwargs]).T) for k in range(1,order+1-N)] + [lambda *kwargs, k=k: np.cos(k*np.array([kwargs]).T) for k in range(N+1)]
    elif expanMethod == 'dirichlet':
        basis =  [ lambda *kwargs, k=k: 1 / ( k ** np.array([kwargs]).T ) for k in range(1, order + 2) ]
        return exactErrorGenerall( basis, fun, x0, x1, dirichlet=True )
    elif expanMethod == 'laurent':
        basis = [ lambda *kwargs, k=k: 1 / (np.array([kwargs]).T**k) for k in range( 1, N+1 ) ] + [ lambda *kwargs, k=k: np.array([kwargs]).T**k for k in range( order+1-N ) ]

    return exactErrorGenerall( basis, fun, x0, x1 )



def exactErrorGenerall( basis, fun, x0, x1, dirichlet=False ):
    order = len( basis ) - 1 # -1 for consistancy with polynomial case, where len(basis) = highest order + 1

    ## Gramian(v1, ..., vn)
    gramianV = np.ones(( order+1, order+1 ))

    if dirichlet and len(x0) == 1:
        for i in range( order + 1 ):
            for j in range( i + 1 ):
                arg, a, b = (i+1) * (j+1), x0[0], x1[0] # rename variables
                
                gramianV[i,j] = (b-a) if arg==1 else (arg ** b - arg ** a) / (np.log(arg) * (arg ** (a+b))) # already integrated
                gramianV[j,i] = gramianV[i,j]
    else:
        for i in range( order + 1 ):
            for j in range( i + 1 ):
                gramianV[i,j] = L2Product( basis[i], basis[j], x0, x1 )
                gramianV[j,i] = gramianV[i,j]
    
    ## vector [(g,v1), ..., (g,vn)]
    vectorWithG = np.zeros( order + 2 )
    vectorWithG[0] = L2Product( fun, fun, x0, x1 )
    for i in range( order + 1 ):
        vectorWithG[i+1] = L2Product( fun, basis[i], x0, x1 )

    ## Gramian(g, v1, ..., vn) - glue the above
    addedLeft = np.concatenate( (vectorWithG.reshape(-1,1)[1:], gramianV), axis=1 )
    gramianVWithG = np.concatenate(( vectorWithG.reshape(1,-1), addedLeft ))

    ## calculate detG(g, v1, ..., vn) and return
    detGramianV = np.linalg.det( gramianV )
    detGramianVWithG = np.linalg.det( gramianVWithG )

    if ( detGramianV <= 0 or detGramianVWithG < 0): print( "WARNING: detGramianV or detGramianVWithG are <= 0" )
    return np.inf if detGramianV == 0 else ( np.sqrt( np.abs(detGramianVWithG) ) / np.sqrt( np.abs(detGramianV) ) )