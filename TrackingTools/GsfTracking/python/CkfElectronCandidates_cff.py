import FWCore.ParameterSet.Config as cms

# initialize magnetic field #########################
# initialize geometry #####################

# KFUpdatoerESProducer
from TrackingTools.KalmanUpdators.KFUpdatorESProducer_cfi import *
# Chi2MeasurementEstimatorESProducer
#include "TrackingTools/KalmanUpdators/data/Chi2MeasurementEstimatorESProducer.cfi"
# KFTrajectoryFitterESProducer
#include "TrackingTools/TrackFitters/data/KFTrajectoryFitterESProducer.cfi"
# KFTrajectorySmootherESProducer
#include "TrackingTools/TrackFitters/data/KFTrajectorySmootherESProducer.cfi"
# KFFittingSmootherESProducer
#include "TrackingTools/TrackFitters/data/KFFittingSmootherESProducer.cfi"
# PropagatorWithMaterialESProducer
#include "TrackingTools/MaterialEffects/data/MaterialPropagator.cfi"
# PropagatorWithMaterialESProducer
#include "TrackingTools/MaterialEffects/data/OppositeMaterialPropagator.cfi"
# stripCPE
from RecoLocalTracker.SiStripRecHitConverter.StripCPEfromTrackAngle_cfi import *
from RecoLocalTracker.SiStripRecHitConverter.SiStripRecHitMatcher_cfi import *
# pixelCPE
from RecoLocalTracker.SiPixelRecHits.PixelCPEParmError_cfi import *
#TransientTrackingBuilder
from RecoTracker.TransientTrackingRecHit.TransientTrackingRecHitBuilder_cfi import *
from RecoTracker.CkfPattern.CkfTrackCandidates_cff import *
# generate CTF track candidates ############
from TrackingTools.GsfTracking.CkfElectronCandidatesChi2_cfi import *
from TrackingTools.GsfTracking.FwdElectronPropagator_cfi import *
from TrackingTools.GsfTracking.BwdElectronPropagator_cfi import *
from TrackingTools.GsfTracking.CkfElectronTrajectoryBuilder_cfi import *
from TrackingTools.GsfTracking.CkfElectronCandidates_cfi import *

