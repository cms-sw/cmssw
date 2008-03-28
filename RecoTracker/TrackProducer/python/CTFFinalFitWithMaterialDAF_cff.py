import FWCore.ParameterSet.Config as cms

# magnetic field
from MagneticField.Engine.volumeBasedMagneticField_cfi import *
# cms geometry
#include "Geometry/TrackerRecoData/data/trackerRecoGeometryXML.cfi"
from Geometry.CMSCommonData.cmsIdealGeometryXML_cfi import *
# tracker geometry
from Geometry.TrackerGeometryBuilder.trackerGeometry_cfi import *
# tracker numbering
from Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi import *
# KFUpdatoerESProducer
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
# Chi2MeasurementEstimatorESProducer
#include "TrackingTools/KalmanUpdators/data/Chi2MeasurementEstimatorESProducer.cfi"
# Chi2MeasurementEstimatorESProducer this is used by the fitting-smoother
from TrackingTools.KalmanUpdators.MRHChi2MeasurementEstimatorESProducer_cfi import *
import copy
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
# PropagatorWithMaterialESProducer
RungeKuttaTrackerPropagator = copy.deepcopy(MaterialPropagator)
# KFTrajectoryFitterESProducer
from TrackingTools.TrackFitters.DAFTrajectoryFitterESProducer_cfi import *
# KFTrajectorySmootherESProducer
from TrackingTools.TrackFitters.DAFTrajectorySmootherESProducer_cfi import *
# KFFittingSmootherESProducer
from TrackingTools.TrackFitters.DAFFittingSmootherESProducer_cfi import *
#MultiMeasurementTracker
from RecoTracker.SiTrackerMRHTools.GroupedMultiRecHitCollector_cff import *
from RecoTracker.SiTrackerMRHTools.SimpleMultiRecHitCollector_cff import *
#multiRecHitUpdator
from RecoTracker.SiTrackerMRHTools.SiTrackerMultiRecHitUpdator_cff import *
# PropagatorWithMaterialESProducer
#include "TrackingTools/MaterialEffects/data/OppositeMaterialPropagator.cfi"
# stripCPE
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
# pixelCPE
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
# TrackProducer
from RecoTracker.TrackProducer.CTFFinalFitWithMaterialDAF_cfi import *
RungeKuttaTrackerPropagator.ComponentName = 'RungeKuttaTrackerPropagator'
RungeKuttaTrackerPropagator.useRungeKutta = True

