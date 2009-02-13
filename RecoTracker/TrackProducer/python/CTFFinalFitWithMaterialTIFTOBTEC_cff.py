import FWCore.ParameterSet.Config as cms

# magnetic field

from MagneticField.Engine.uniformMagneticField_cfi import *
# cms geometry

# tracker geometry
# tracker numbering
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
KFFittingSmootherTIFTOBTEC = copy.deepcopy(KFFittingSmoother)
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
ctfWithMaterialTracksTIFTOBTEC = copy.deepcopy(ctfWithMaterialTracks)
KFFittingSmootherTIFTOBTEC.ComponentName = 'KFFittingSmootherTIFTOBTEC'
KFFittingSmootherTIFTOBTEC.MinNumberOfHits = 4
ctfWithMaterialTracksTIFTOBTEC.src = 'ckfTrackCandidatesTIFTOBTEC'
ctfWithMaterialTracksTIFTOBTEC.Fitter = 'KFFittingSmootherTIFTOBTEC'

