import FWCore.ParameterSet.Config as cms

import TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi 
#
# "backward" propagator for electrons
#
bwdElectronPropagator = TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi.OppositeMaterialPropagator.clone(
    Mass          = 0.000511,
    ComponentName = 'bwdElectronPropagator'
)
