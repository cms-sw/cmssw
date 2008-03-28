import FWCore.ParameterSet.Config as cms

# magnetic field
#include "Geometry/CMSCommonData/data/cmsMagneticFieldXML.cfi"
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
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
# KFTrajectoryFitterESProducer
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
# KFTrajectorySmootherESProducer
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
import copy
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
# KFFittingSmootherESProducer
#include "TrackingTools/TrackFitters/data/KFFittingSmootherESProducer.cfi"
KFFittingSmootherP5 = copy.deepcopy(KFFittingSmoother)
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
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
# TrackProducer
ctfWithMaterialTracksP5 = copy.deepcopy(ctfWithMaterialTracks)
KFFittingSmootherP5.ComponentName = 'KFFittingSmootherP5'
KFFittingSmootherP5.MinNumberOfHits = 4
ctfWithMaterialTracksP5.src = 'ckfTrackCandidatesP5'
ctfWithMaterialTracksP5.Fitter = 'KFFittingSmootherP5'

