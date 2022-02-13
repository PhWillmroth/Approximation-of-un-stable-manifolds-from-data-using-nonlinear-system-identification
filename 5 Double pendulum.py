#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('.\\data')
sys.path.append('.\\utils')
sys.path.append('.\\Sindy')
sys.path.append('.\\Expansion')
sys.path.append('.\\plot')

# ------------------------- Double pendulum -----------------------------

from data.doubPendDopri import doubPendDopri
from Sindy.doublePendulum import doublePendulum, getDpRank
from Expansion.doubPendExpansion import doubPendExpan
import plot.plotDoublePendulum as pdp
from parameters import doubPend as dp

## GENERATE DATA
doubPendDopri( '' ) # exact data
doublePendulum( '' , '' ) # sindy
doubPendExpan( '' , '' ) # expan

## PLOT
# plot curve
pdp.plotDoubPendCurve( '', '', show=True )
pdp.plotDoubPendCurve( '', '', show=True, includeExpan=True )

# plot states
pdp.plotDoubPendState( '', '', show=True )
pdp.plotDoubPendState( '', '', show=True, includeExpan=True )

# plot errors
pdp.plotDoubPendError( '', [f'Sindy{dp.tEnd}_{eta}' for eta in [10, 1, 0.1, 0.01, 0.001, 0.0001]], 
                            ['10', '1', '10^{-1}', '10^{-2}', '10^{-3}', '10^{-4}'], show=True )



