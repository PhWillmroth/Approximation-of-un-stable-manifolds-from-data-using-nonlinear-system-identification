#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('.\\data')
sys.path.append('.\\utils')
sys.path.append('.\\Sindy')
sys.path.append('.\\Expansion')
sys.path.append('.\\plot')

#2D Linear Oscillator
from data.linOscExact import linOscExact
from data.linOscDopri import linOscDopri
from Sindy.linearOscillator import linearOscillator
from Expansion.linOscExpansion import linOscExpan
from plot.plotLinearOscillator import plotLinearOscillator
from parameters import linOsc as lo

linOscExact( '' ) # exact data
linOscDopri( '' ) # 'exact' data
linearOscillator( '', '' )
linOscExpan('', '')
plotLinearOscillator( '', '', show=True )
plotLinearOscillator( '', '', includeExpan=True, show=True )