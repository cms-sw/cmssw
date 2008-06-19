import FWCore.ParameterSet.Config as cms

from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
from TrackingTools.MaterialEffects.Propagators_cff import *
from TrackingTools.TrackFitters.TrackFitters_cff import *
import TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi
FittingSmootherRKP5 = TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi.KFFittingSmoother.clone()
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
import RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi
ctfWithMaterialTracksP5 = RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi.ctfWithMaterialTracks.clone()
FittingSmootherRKP5.ComponentName = 'FittingSmootherRKP5'
FittingSmootherRKP5.Fitter = 'RKFitter'
FittingSmootherRKP5.Smoother = 'RKSmoother'
FittingSmootherRKP5.MinNumberOfHits = 4
ctfWithMaterialTracksP5.src = 'ckfTrackCandidatesP5'
ctfWithMaterialTracksP5.Fitter = 'FittingSmootherRKP5'


