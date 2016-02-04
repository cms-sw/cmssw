import FWCore.ParameterSet.Config as cms


from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *

# stripCPE
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
# pixelCPE
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *

# TrackProducer
import RecoTracker.TrackProducer.TrackProducer_cfi 
ctfPixelLess = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src = cms.InputTag("ckfTrackCandidatesPixelLess"),  
    Fitter = cms.string('RKFittingSmoother')
)

