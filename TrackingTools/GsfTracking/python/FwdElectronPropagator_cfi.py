import FWCore.ParameterSet.Config as cms

import copy
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
#
# "forward" propagator for electrons
#
fwdElectronPropagator = copy.deepcopy(MaterialPropagator)
fwdElectronPropagator.Mass = 0.000511
fwdElectronPropagator.ComponentName = 'fwdElectronPropagator'

