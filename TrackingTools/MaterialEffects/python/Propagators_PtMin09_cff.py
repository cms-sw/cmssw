import FWCore.ParameterSet.Config as cms

# Define propagators that take into account the uncertainty on the 
# reconstructed track momentum, when estimating the rms scattering
# angle.

# When doing this, the track Pt is assumed not to be below ptMin, 
# to avoid large scattering angles.

# This propagator may useful during the track building phase,
# but should probably not be used for track fitting.

import TrackingTools.MaterialEffects.MaterialPropagator_cfi
MaterialPropagatorPtMin09 = TrackingTools.MaterialEffects.MaterialPropagator_cfi.MaterialPropagator.clone(
    ComponentName = 'PropagatorWithMaterialPtMin09',
    ptMin         = 0.9
)
import TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi
OppositeMaterialPropagatorPtMin09 = TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi.OppositeMaterialPropagator.clone(
    ComponentName = 'PropagatorWithMaterialOppositePtMin09',
    ptMin         = 0.9
)
