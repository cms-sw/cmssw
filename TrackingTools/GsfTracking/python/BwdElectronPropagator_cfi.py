import FWCore.ParameterSet.Config as cms

import copy
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
#
# "backward" propagator for electrons
#
bwdElectronPropagator = copy.deepcopy(OppositeMaterialPropagator)
bwdElectronPropagator.Mass = 0.000511
bwdElectronPropagator.ComponentName = 'bwdElectronPropagator'

