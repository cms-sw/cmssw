import FWCore.ParameterSet.Config as cms

#special propagator
from TrackingTools.GeomPropagators.BeamHaloPropagator_cff import *
from TrackingTools.TrackFitters.KFTrajectoryFitter_cfi import *
# KFTrajectoryFitterESProducer
#include "TrackingTools/TrackFitters/data/KFTrajectoryFitterESProducer.cfi"
KFTrajectoryFitterBeamHalo = KFTrajectoryFitter.clone(
    ComponentName = 'KFFitterBH',
    Propagator    = 'BeamHaloPropagatorAlong'
)
from TrackingTools.TrackFitters.KFTrajectorySmoother_cfi import *
# KFTrajectorySmootherESProducer
#include "TrackingTools/TrackFitters/data/KFTrajectorySmootherESProducer.cfi"
KFTrajectorySmootherBeamHalo = KFTrajectorySmoother.clone(
    ComponentName = 'KFSmootherBH',
    Propagator    = 'BeamHaloPropagatorAlong'
)
from TrackingTools.TrackFitters.KFFittingSmoother_cfi import *
# KFFittingSmootherESProducer
#include "TrackingTools/TrackFitters/data/KFFittingSmootherESProducer.cfi"
KFFittingSmootherBeamHalo = KFFittingSmoother.clone(
    ComponentName = 'KFFittingSmootherBH',
    Fitter        = 'KFFitterBH',
    Smoother      = 'KFSmootherBH'
)
# generate the final tracks ######################
#get the dependencies
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
from RecoTracker.TrackProducer.TrackProducer_cfi import *
#clone the track producer
beamhaloTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
    src                 = 'beamhaloTrackCandidates',
    Fitter              = 'KFFittingSmootherBH',
    Propagator          = 'BeamHaloPropagatorAlong',
    TTRHBuilder         = 'WithTrackAngle',
    NavigationSchool    = 'BeamHaloNavigationSchool',
    AlgorithmName       = 'beamhalo',
    alias               = 'beamhaloTracks',
    GeometricInnerState = True
)
