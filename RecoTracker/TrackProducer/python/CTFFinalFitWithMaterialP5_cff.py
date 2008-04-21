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
from TrackingTools.KalmanUpdators.Chi2MeasurementEstimatorESProducer_cfi import *
import copy
from TrackingTools.MaterialEffects.MaterialPropagator_cfi import *
# PropagatorWithMaterialESProducer
RungeKuttaTrackerPropagator = copy.deepcopy(MaterialPropagator)
import copy
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
# KFTrajectoryFitterESProducer
FitterRK = copy.deepcopy(KFTrajectoryFitter)
import copy
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
# KFTrajectorySmootherESProducer
SmootherRK = copy.deepcopy(KFTrajectorySmoother)
import copy
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
# KFFittingSmootherESProducer
FittingSmootherRKP5 = copy.deepcopy(KFFittingSmoother)
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
RungeKuttaTrackerPropagator.ComponentName = 'RungeKuttaTrackerPropagator'
RungeKuttaTrackerPropagator.useRungeKutta = True
FitterRK.ComponentName = 'FitterRK'
FitterRK.Propagator = 'RungeKuttaTrackerPropagator'
SmootherRK.ComponentName = 'SmootherRK'
SmootherRK.Propagator = 'RungeKuttaTrackerPropagator'
FittingSmootherRKP5.ComponentName = 'FittingSmootherRKP5'
FittingSmootherRKP5.Fitter = 'FitterRK'
FittingSmootherRKP5.Smoother = 'SmootherRK'
FittingSmootherRKP5.MinNumberOfHits = 4
ctfWithMaterialTracksP5.src = 'ckfTrackCandidatesP5'
ctfWithMaterialTracksP5.Fitter = 'FittingSmootherRKP5'

