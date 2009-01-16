import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackRefitter.TracksToTrajectories_cff import *
# IMPORTANT !!! ##
# If you want to revert the fit direction, then
# Case 1 #
# string RefitDirection = "alongMomentum"
# KFTrajectoryFitterESProducer   ---> Fitter = "KFFitterForRefitInsideOut"
# KFTrajectorySmootherESProducer ---> Smoother = "KFSmootherForRefitInsideOut"
# Case 2 #
# string RefitDirection = "alongMomentum"
# KFTrajectoryFitterESProducer   ---> Fitter = "KFFitterForRefitOutsideIn"
# KFTrajectorySmootherESProducer ---> Smoother = "KFSmootherForRefitOutsideIn"
# Case 3 #
# string RefitDirection = "oppositeToMomentum"
# KFTrajectoryFitterESProducer   ---> Fitter = "KFFitterForRefitOutsideIn"
# KFTrajectorySmootherESProducer ---> Smoother = "KFSmootherForRefitOutsideIn"
# Case 4 #
# string RefitDirection = "oppositeToMomentum"
# KFTrajectoryFitterESProducer   ---> Fitter = "KFFitterForRefitOutsideIn"
# KFTrajectorySmootherESProducer ---> Smoother = "KFSmootherForRefitOutsideIn"
#
globalCosmicMuons = cms.EDProducer("TracksToTrajectories",
                                   Type = cms.string("CosmicMuonsForAlignment"),
                                   Tracks = cms.InputTag("cosmicMuons"),
                                   TrackTransformer = cms.PSet(TrackerRecHitBuilder = cms.string('WithTrackAngle'),
                                                               MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
                                                               RefitRPCHits = cms.bool(True),
                                                               )
                                   )



MuAlGlobalCosmics = globalCosmicMuons.clone()
MuAlGlobalCosmics.Tracks = cms.InputTag("ALCARECOMuAlGlobalCosmics","GlobalMuon")
