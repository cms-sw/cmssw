import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackRefitter.TracksToTrajectories_cff import *
# IMPORTANT !!! ##
# If you want to revert the fit direction, then
# Case 1 #
# string RefitDirection = "insideOut"
# KFTrajectoryFitterESProducer   ---> Fitter = "KFFitterForRefitInsideOut"
# KFTrajectorySmootherESProducer ---> Smoother = "KFSmootherForRefitInsideOut"
# Case 2 #
# string RefitDirection = "outsideIn"
# KFTrajectoryFitterESProducer   ---> Fitter = "KFFitterForRefitOutsideIn"
# KFTrajectorySmootherESProducer ---> Smoother = "KFSmootherForRefitOutsideIn"
# the propagator must be the same as the one used by the Fitter
#
globalMuons = cms.EDFilter("TracksToTrajectories",
    Tracks = cms.InputTag("globalMuons"),
    TrackTransformer = cms.PSet(
        Fitter = cms.string('KFFitterForRefitInsideOut'),
        TrackerRecHitBuilder = cms.string('WithTrackAngle'),
        Smoother = cms.string('KFSmootherForRefitInsideOut'),
        MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
        RefitDirection = cms.string('insideOut'),
        RefitRPCHits = cms.bool(True),
        Propagator = cms.string('SmartPropagatorAny')
    )
)


