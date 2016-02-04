import FWCore.ParameterSet.Config as cms

import TrackingTools.MaterialEffects.MaterialPropagator_cfi
# "forward" propagator for electrons
fwdGsfElectronPropagator = TrackingTools.MaterialEffects.MaterialPropagator_cfi.MaterialPropagator.clone()
fwdGsfElectronPropagator.Mass = 0.000511
fwdGsfElectronPropagator.ComponentName = 'fwdGsfElectronPropagator'

