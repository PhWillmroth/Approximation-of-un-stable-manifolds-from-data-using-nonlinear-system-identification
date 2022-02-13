#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import ode
from scipy.constants import g as GRAV

class Functions:
    # time paramters data
    tStart = 0
    tStep = 0.1

    # time paramters training
    trainBatchSize = 0.5 # take 50% of the genereated data to train the model; and the other 50% for validation

    # Sindy parameters
    functionDict = [] # use different functions as candidates for Sindy-model

    # Series Expansion
    expanMethod = 'chebyshev'
    partSumOrder = 7 # order of the partial sum (cut off)

    # RK tolerance parameters 
    relTol = 1e-10 
    absTol = 1e-10


    # eta: noise variance on derivative
    # x0: initial condition
    # lmbd: sparsification knob for Sindy
    # polyorder: highest order of included polynoms in Theta (Sindy)
    # unfortunately sklearn.preprocessing.PolynomialFeatures does not accept polyorder 0; for polyorder 1 it returns [1, x1, x2, ..., xn]
    

    ## for shorter notation making random numbers
    def randn( a ):
        return np.random.standard_normal( size=a )

    ## get the state dimension automatically, so I wont have to pass it
    def n( self ):
        return int( self.x0.size )

    
    ## Solve ODE using runge-kutta
    tSub = []
    def solveOde( self, rhs, getTimeStep=False, myMethod='dopri5', **kwargs ):

        # init integrator
        solver = ode( rhs )                                     
        solver.set_integrator( myMethod, atol=self.absTol, rtol=self.relTol )
        solver.set_initial_value( self.x0, self.tStart )        # set initial values
        solver.set_solout( lambda t, y: tSub.append(t) )        # this is called every time 'solver' was succesfull;
                                                                # it saves number of subtimesteps;

        # set intergration parameters                                                   
        [tStart,tEnd,tStep] = kwargs.get( 'tOde', [self.tStart,self.tEnd,self.tStep] ) # get time
        t = np.arange( tStart, tEnd, tStep )                    # init vector of sampling times (s.o.)
        x = np.empty(( t.size, self.n(self) ))                  # init x as empty (tCount,n)-array
        x[0] = self.x0                                          # first entry of x := innitial condition
        k = 1                                                   # a counter
        rkSteps = np.empty( t.size )                            # saves the number of subtimesteps dopri takes
        
        # integrate ODE
        lenT = t.size
        while solver.successful() and k < lenT and solver.t < (tEnd - tStep):
            tSub = []
            solver.integrate( t[k] )
            x[k] = solver.y
            rkSteps[k] = len( tSub )
            k += 1

        # output
        if getTimeStep:
            return x, rkSteps
        else:
            return x



# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

class linOsc ( Functions ):
    eta = .05
    tEnd = 50
    x0 = np.array([ 2, 0 ])  # is hardcoded in linOscexact (-> part of the exact solution)
    polyorder = 5
    lmbd = .039
    expanMethod = 'legendre'
    partSumOrder = 1
    dOmg1, dOmg2 = [-0.2,]*2, [0.2,]*2

# ------------------------------------------------------------------------------------
class cubOsc( Functions ):

    eta = .05
    tEnd = 50
    x0 = np.array([ 2, 0 ])
    polyorder = 5
    lmbd = .04
    expanMethod = 'legendre'
    partSumOrder = 3
    dOmg1, dOmg2 = [0,]*2, [25,]*2


# ------------------------------------------------------------------------------------
class dddOsc( Functions ):

    eta = .01
    tEnd = 25
    x0 = np.array([ 2, 0, 1 ])
    polyorder = 2                   # Increasing polyorder gives way worse results
    lmbd = 0.085
    expanMethod = 'legendre'
    partSumOrder = 1
    dOmg1, dOmg2 = [-0.6,-0.6,0.01], [0.6, 0.6, 0.03]


# ------------------------------------------------------------------------------------
class Lorenz( Functions ):
    
    # time paramters 'exact'
    tStart, tEnd, tStep = 0, 100, .001

    # time parameters sindy solve
    tSindyStart, tSindyEnd, tSindyStep = 0, 250, .001

    # time paramters plot
    tPlotStart, tPlotEnd, tPlotStep = 0.001, 20., .001


    eta = 0.01

    # Lorenz System parameters
    sigma, beta, rho = 10, 8/3, 28
    
    # inital condition
    x0 = np.array([ -8, 7, 27 ])

    # Sindy parameters
    polyorder =  3
    lmbd = 0.025

    # Expan parameters
    expanMethod = 'power'
    partSumOrder = 2
    
    # RK parameters 
    relTol = 1e-12          # smaller than before
    absTol = 1e-12

    # Omega over which is integrated for exact error and L2-norm
    dOmg1, dOmg2 = [-20, -50, -5], [20, 50, 60]

    

# ------------------------------------------------------------------------------------
class doubPend( Functions ):
    
    eta = .01

    # time paramters dopri/training
    tStart, tEnd, tStep = 0, 8, .001

    # time parameters sindy solve
    tSindyStart, tSindyEnd, tSindyStep = 0, 8, .001

    # time paramters plot
    tPlotStart, tPlotEnd, tPlotStep = 0, 8, .001

    # inital condition
    x0 = np.array([ np.pi/2, 0, 0, 0 ])

    # model paramters (only for the plot of an example pendulum; change also manually in doubPendDopri)
    l1, l2 = 1, 1

    # Sindy parameters
    polyorder =  3
    lmbd = .035
    
    # articifial function dictionary
    '''functionDict = [lambda x: ((0.5 * np.sin(2*(x[:,0]-x[:,1])) * (x[:,2]**2)) / ((np.cos(x[:,0]-x[:,1]))**2 - 2))[None].T,
                    lambda x: ((-np.cos(x[:,0]-x[:,1]) * GRAV * np.sin(x[:,1])) / ((np.cos(x[:,0]-x[:,1]))**2 - 2))[None].T,
                    lambda x: ((np.sin(x[:,0]-x[:,1]) * (x[:,3]**2)) / ((np.cos(x[:,0]-x[:,1]))**2 - 2))[None].T,
                    lambda x: ((2 * GRAV * np.sin(x[:,0])) / ((np.cos(x[:,0]-x[:,1]))**2 - 2))[None].T,
                    lambda x: ((-0.5 * np.sin(2*(x[:,0]-x[:,1])) * (x[:,3]**2)) / ((np.cos(x[:,0]-x[:,1]))**2 - 2))[None].T,
                    lambda x: ((-2 * np.sin(x[:,0]-x[:,1]) * (x[:,2]**2)) / ((np.cos(x[:,0]-x[:,1]))**2 - 2))[None].T,
                    lambda x: ((-2 * GRAV * np.cos(x[:,0]-x[:,1]) * np.sin(x[:,0])) / ((np.cos(x[:,0]-x[:,1]))**2 - 2))[None].T,
                    lambda x: ((2 * GRAV * np.sin(x[:,1])) / ((np.cos(x[:,0]-x[:,1]))**2 - 2))[None].T
                    ]'''

    # Expan parameters
    partSumOrder = 2
    expanMethod = 'dirichlet'

    # RK parameters 
    relTol = 1e-12
    absTol = 1e-12

    # Omega over which is integrated for exact error and L2-norm
    dOmg1, dOmg2 = [-20, -50, -5], [20, 50, 60]
    dOmg1, dOmg2 = [-2.5,]*2, [2.5,]*2
# ------------------------------------------------------------------------------------

class pathological( Functions ):
    eta = .05
    tStart = 0
    tEnd = 25
    x0 = np.array([ 0 ])
    polyorder = 5
    lmbd = .039
    expanMethod = 'legendre'
    partSumOrder = 6
    name = 'weierstrassFunction'
    dOmg1, dOmg2 = [tStart,], [tEnd,] 



# --------------------------------- For [Ohlsson et al. 2007]---------------------------
    class spiral( Functions ):

    tStart = 0
    tEnd = 4
    tStep = (tEnd-tStart) / 500

    tTossiStart = 0
    tTossiEnd = 1.5
    eta = .05
    nNeighbours = 10 # initial number of neighbours for LLE
    intrinsicDim = 1 # dimension of the manifold
    
    # Sindy
    polyorder = 5
    lmbd = .05
    method = 'LLE'
    methodList = [ 'LLE', 'MLLE', 'HLLE', 'LTSA', 'Isomap', 'SpectralEmbedding', 'MDS', 'TSNE' ]





