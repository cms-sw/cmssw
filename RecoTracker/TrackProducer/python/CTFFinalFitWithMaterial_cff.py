import FWCore.ParameterSet.Config as cms

from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
from TrackingTools.KalmanUpdators.KFSwitching1DUpdatorESProducer_cfi import *
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
from TrackingTools.MaterialEffects.Propagators_cff import *
from TrackingTools.TrackFitters.TrackFitters_cff import *
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoTracker.TransientTrackingRecHit.TTRHBuilders_cff import *
from RecoTracker.TrackProducer.TrackProducer_cfi import *
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *

