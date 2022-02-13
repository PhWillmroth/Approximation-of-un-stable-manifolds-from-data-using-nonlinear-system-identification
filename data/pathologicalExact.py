## generate exact data of a pathological function
import numpy as np
import pandas as pd
from scipy.special import zeta

from parameters import pathological as pa

center = pa.tEnd / 4
center2 = pa.tEnd / 2
periodicity = pa.tEnd / 5

def xSinOneOverX( x ):
    s = np.copy( x )
    s = 10 / (s - center )
    s = np.array( s ) if type(s) == np.float64 else s
    s[s==np.inf] = 0
    return s * np.sin( s )

def sinOneOverX( x ):
    s = np.copy( x )
    s = 10 / (s - center )
    s = np.array( s ) if type(s) == np.float64 else s
    s[s==np.inf] = 0
    return np.sin( s )

a = 0.25
b = 29
cutOff = 30
def weierstrassFunction( x ):
    sum = 0
    for i in range( cutOff ):
        sum += a**i * np.cos( b**i * np.pi * x / periodicity )
    return sum

def logLog( x ):
    out = np.log( np.log( 1 + ( 1 / np.abs(x-center2) ) ) )
    out = np.array( out ) if type(out) == np.float64 else out
    out[out==np.inf] = 0
    return out

# Anm.: bei gleicher Funktion ist lambda etwas schneller als def
fDict = { 'constant' : lambda x: x ** 0,
    'floor' : lambda x: np.floor( x / periodicity ),
    'signum' : lambda x: np.sign( x - center ),
    'absoluteValue' : lambda x: np.abs( x - center2 ),
    'xSinOneOverX' : xSinOneOverX,
    'sinOneOverX' : sinOneOverX,
    'eToMinusOneOverX' : lambda x: np.exp( -1 / x ),
    'weierstrassFunction' : weierstrassFunction,
    'sawtooth' : lambda x: ( x / periodicity ) - np.floor( x / periodicity ),
    'riemannZeta' : zeta,
    'squareWave' : lambda x: np.sign( np.sin( 2 * np.pi * x / periodicity / periodicity ) ),
    'logLog' : logLog }

# --------------------------------------------------------------------
def pathologicalExact( outName ):
    # prepare stuff
    t = np.arange( pa.tStart, pa.tEnd, pa.tStep )
    x = np.copy( t )
    x.shape = ( x.size, 1 )

    outputData = pd.DataFrame( [], index=t, columns=['x','dx'] ) # init dataframe
    
    pathologicalFunction = fDict[ pa.name ]

    # generate data
    outputData[[ 'x' ]] = x # assign x values

    ## compute function output
    outputData[[ 'dx' ]] = pathologicalFunction( x )

    # save data
    outputData.to_csv( f'.\\data\\pathologicalExact{pa.name}{outName}.csv' )

    # plt.plot(x, pathologicalFunction( x ))
    # plt.title(pa.name)
    # plt.show()



