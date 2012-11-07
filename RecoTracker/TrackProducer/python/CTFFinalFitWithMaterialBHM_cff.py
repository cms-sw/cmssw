import FWCore.ParameterSet.Config as cms

#special propagator
from TrackingTools.GeomPropagators.BeamHaloPropagator_cff import *
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
# KFTrajectoryFitterESProducer
#include "TrackingTools/TrackFitters/data/KFTrajectoryFitterESProducer.cfi"
KFTrajectoryFitterBeamHalo = copy.deepcopy(KFTrajectoryFitter)
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
# KFTrajectorySmootherESProducer
#include "TrackingTools/TrackFitters/data/KFTrajectorySmootherESProducer.cfi"
KFTrajectorySmootherBeamHalo = copy.deepcopy(KFTrajectorySmoother)
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
# KFFittingSmootherESProducer
#include "TrackingTools/TrackFitters/data/KFFittingSmootherESProducer.cfi"
KFFittingSmootherBeamHalo = copy.deepcopy(KFFittingSmoother)
# generate the final tracks ######################
#get the dependencies
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
from RecoTracker.TrackProducer.TrackProducer_cfi import *
#clone the track producer
beamhaloTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone()
KFTrajectoryFitterBeamHalo.ComponentName = 'KFFitterBH'
KFTrajectoryFitterBeamHalo.Propagator = 'BeamHaloPropagatorAlong'
KFTrajectorySmootherBeamHalo.ComponentName = 'KFSmootherBH'
KFTrajectorySmootherBeamHalo.Propagator = 'BeamHaloPropagatorAlong'
KFFittingSmootherBeamHalo.ComponentName = 'KFFittingSmootherBH'
KFFittingSmootherBeamHalo.Fitter = 'KFFitterBH'
KFFittingSmootherBeamHalo.Smoother = 'KFSmootherBH'
beamhaloTracks.src = 'beamhaloTrackCandidates'
beamhaloTracks.Fitter = 'KFFittingSmootherBH'
beamhaloTracks.Propagator = 'BeamHaloPropagatorAlong'
beamhaloTracks.TTRHBuilder = 'WithTrackAngle'
beamhaloTracks.NavigationSchool = 'BeamHaloNavigationSchool'
beamhaloTracks.AlgorithmName = 'beamhalo'
beamhaloTracks.alias = 'beamhaloTracks'
beamhaloTracks.GeometricInnerState = True
