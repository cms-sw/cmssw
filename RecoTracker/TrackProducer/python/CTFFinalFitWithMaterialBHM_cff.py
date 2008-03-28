import FWCore.ParameterSet.Config as cms

#special propagator
from TrackingTools.GeomPropagators.BeamHaloPropagator_cff import *
import copy
from TrackingTools.TrackFitters.KFTrajectoryFitterESProducer_cfi import *
# KFTrajectoryFitterESProducer
#include "TrackingTools/TrackFitters/data/KFTrajectoryFitterESProducer.cfi"
KFTrajectoryFitterBeamHalo = copy.deepcopy(KFTrajectoryFitter)
import copy
from TrackingTools.TrackFitters.KFTrajectorySmootherESProducer_cfi import *
# KFTrajectorySmootherESProducer
#include "TrackingTools/TrackFitters/data/KFTrajectorySmootherESProducer.cfi"
KFTrajectorySmootherBeamHalo = copy.deepcopy(KFTrajectorySmoother)
import copy
from TrackingTools.TrackFitters.KFFittingSmootherESProducer_cfi import *
# KFFittingSmootherESProducer
#include "TrackingTools/TrackFitters/data/KFFittingSmootherESProducer.cfi"
KFFittingSmootherBeamHalo = copy.deepcopy(KFFittingSmoother)
# generate the final tracks ######################
#get the dependencies
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cff import *
import copy
from RecoTracker.TrackProducer.CTFFinalFitWithMaterial_cfi import *
#clone the track producer
ctfWithMaterialTracksBeamHaloMuon = copy.deepcopy(ctfWithMaterialTracks)
KFTrajectoryFitterBeamHalo.ComponentName = 'KFFitterBH'
KFTrajectoryFitterBeamHalo.Propagator = 'BeamHaloPropagatorAlong'
KFTrajectorySmootherBeamHalo.ComponentName = 'KFSmootherBH'
KFTrajectorySmootherBeamHalo.Propagator = 'BeamHaloPropagatorAlong'
KFFittingSmootherBeamHalo.ComponentName = 'KFFittingSmootherBH'
KFFittingSmootherBeamHalo.Fitter = 'KFFitterBH'
KFFittingSmootherBeamHalo.Smoother = 'KFSmootherBH'
ctfWithMaterialTracksBeamHaloMuon.Fitter = 'KFFittingSmootherBH'
ctfWithMaterialTracksBeamHaloMuon.Propagator = 'BeamHaloPropagatorAlong'
ctfWithMaterialTracksBeamHaloMuon.src = 'ckfTrackCandidatesBeamHaloMuon'
ctfWithMaterialTracksBeamHaloMuon.AlgorithmName = 'beamhalo'

