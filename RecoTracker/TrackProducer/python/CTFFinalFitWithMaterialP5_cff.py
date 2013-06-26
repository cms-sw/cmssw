import FWCore.ParameterSet.Config as cms

from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
from TrackingTools.MaterialEffects.Propagators_cff import *
from TrackingTools.TrackFitters.TrackFitters_cff import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *


#could it just use the standard KFFittingSmootherWithOutliersRejectionAndRK ??
import TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi
FittingSmootherRKP5 = TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi.KFFittingSmoother.clone()
FittingSmootherRKP5.ComponentName = 'FittingSmootherRKP5'
FittingSmootherRKP5.Fitter = 'RKFitter'
FittingSmootherRKP5.Smoother = 'RKSmoother'
FittingSmootherRKP5.MinNumberOfHits = 4 #why is this set to 4??
FittingSmootherRKP5.EstimateCut = cms.double(20.0)
FittingSmootherRKP5.BreakTrajWith2ConsecutiveMissing = cms.bool(False)

import RecoTracker.TrackProducer.TrackProducer_cfi
ctfWithMaterialTracksCosmics = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = 'ckfTrackCandidatesP5',
    Fitter = 'FittingSmootherRKP5',
    #TTRHBuilder = 'WithTrackAngle',
    AlgorithmName = cms.string('ctf'),
    GeometricInnerState = cms.bool(True)
)
