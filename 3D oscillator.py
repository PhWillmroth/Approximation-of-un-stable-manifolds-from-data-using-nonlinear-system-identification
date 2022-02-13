#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('.\\data')
sys.path.append('.\\utils')
sys.path.append('.\\Sindy')
sys.path.append('.\\Expansion')
sys.path.append('.\\plot')

## 3D Linear Oscillator
from data.dddOscDopri import dddOscDopri
from Sindy.dddOscillator import dddOscillator
from Expansion.dddOscExpansion import dddOscExpan
from plot.plotdddOscillator import plotdddOscillator
from parameters import dddOsc

dddOscDopri( '' )
dddOscillator( '', '' )
dddOscExpan( '', '' )
plotdddOscillator( '', '', show=True )
plotdddOscillator( '', '', includeExpan=True, show=True )






