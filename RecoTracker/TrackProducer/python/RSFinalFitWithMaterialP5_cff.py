import FWCore.ParameterSet.Config as cms

# magnetic field
# cms geometry
# tracker geometry
# tracker numbering
# KFUpdatoerESProducer
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
# Chi2MeasurementEstimatorESProducer
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
# KFTrajectoryFitterESProducer
# KFTrajectorySmootherESProducer
# KFFittingSmootherESProducer
from TrackingTools.TrackFitters.RungeKuttaFitters_cff import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
# PropagatorWithMaterialESProducer
from TrackingTools.MaterialEffects.OppositeMaterialPropagator_cfi import *
# stripCPE
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
# pixelCPE
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *

import RecoTracker.TrackProducer.RSFinalFitWithMaterial_cff
# include TrackProducer and clone with new module label
rsWithMaterialTracksCosmics = RecoTracker.TrackProducer.RSFinalFitWithMaterial_cff.rsWithMaterialTracks.clone(
    src = 'rsTrackCandidatesP5',
    TTRHBuilder = 'WithTrackAngle'
    )

