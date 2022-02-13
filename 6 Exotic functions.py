#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('.\\data')
sys.path.append('.\\utils')
sys.path.append('.\\Sindy')
sys.path.append('.\\Expansion')
sys.path.append('.\\plot')

import numpy as np
from utils.collectData import *
from utils.exactError import *
from data.pathologicalExact import fDict



# ------------------------- pathological function -----------------------------
from data.pathologicalExact import pathologicalExact
from Sindy.pathologicalFunction import pathologicalFunction
from Expansion.pathologicalExpansion import pathologicalExpan
import plot.plotPathological as ppa
from parameters import pathological as pa


pathologicalExact("")
pathologicalFunction("","")
pathologicalExpan("","")
ppa.plotPathologicalFunction("","","","")




REDUNDANT = 500
pa.trainBatchSize = 1
for pa.name, pa.expanMethod, pa.tStep in [
    ('signum', 'chebyshev', 5.0 ),
    ('absoluteValue', 'power', 4.0),
    ('eToMinusOneOverX', 'dirichlet', 1.5385307692307693),
    ('sawtooth', 'legendre', 0.0001),
    ('squareWave', 'fourier', 6.0),
    ('logLog', 'legendre', 5.0)
    ]:
    pa.polyorder = 5
    pa.partSumOrder = 5
    for pa.tStep in [0.0001,] + list(np.arange(0.125, 10.125, .125)):
    # Alternative: for pa.polyorder in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        pa.partSumOrder = pa.polyorder
        pathologicalExact( f'step{pa.tStep}' )

        for pa.eta in [0.0001, 0.001, 0.01, 0.1, 1, 10]:
            for pa.lmbd in [0.001, 0.01, 0.04, 0.06, 0.08, 1]:

                # SINDY
                sum1 = np.zeros(2)
                for i in range( REDUNDANT ):
                    sum1 += pathologicalFunction( f'step{pa.tStep}', f'step{pa.tStep}' )
                [e2, e3] = sum1 / REDUNDANT
                e1 = exactErrorSindy( pa, fDict.get(pa.name), pa.polyorder )

                # EXPAN
                sum2 = np.zeros(2)
                for k in range( REDUNDANT ):
                    sum2 += pathologicalExpan( f'step{pa.tStep}', f'step{pa.tStep}' ) 
                [e5, e6] = sum2 / REDUNDANT
                e4 = exactErrorExpan( pa, fDict.get(pa.name) )

                saveFindings( pa, 3, L2ErrorExactSindy=e1, L2ErrorSindy=e2, L2ErrorExactExpan=e4, L2ErrorExpan=e5, infErrorSindy=e3, infErrorExpan=e6 )
                print( '#####', pa.name,
                        'L2Error ex/Sindy:', round(e1, 5), round(e2, 5),
                        'L2Error ex/Expan:', round(e4, 5), round(e5, 5),
                        '- order:', pa.partSumOrder)





pa.trainBatchSize = 1
for name in ['signumv5', 'eToMinusOneOverXv2', 'logLogv3', 'sawtoothv2', 'squareWavev2', 'logLog', 'signum', 'absoluteValue',  'eToMinusOneOverX',  'sawtooth',  'squareWave']:
    ppa.plotFindings( f'{name}', 'Sindy', norm='L2', t='polyorder' )
    ppa.plotFindings( f'{name}', 'Expan', norm='L2', t='polyorder' ) pa.trainBatchSize = 1

# The plots need to be done seperately from fitting, b.c. of some python bug: https://github.com/matplotlib/matplotlib/issues/18157
for pa.name in ['signum', 'absoluteValue', 'eToMinusOneOverX', 'sawtooth', 'squareWave', 'logLog']:
    ppa.plotPathologicalFunction('step0.0001', 'step0.0001', expanName='step0.0001', outName='step0.0001', includeExpan=True, show=True )