import FWCore.ParameterSet.Config as cms

from TrackingTools.TrackRefitter.TracksToTrajectories_cff import *
# IMPORTANT !!! ##
# If you want to revert the fit direction, then
# Case 1 #
# string RefitDirection = "alongMomentum"
# KFTrajectoryFitterESProducer   ---> Fitter = "KFFitterForRefitInsideOut"
# KFTrajectorySmootherESProducer ---> Smoother = "KFSmootherForRefitInsideOut"
# Case 2 #
# string RefitDirection = "oppositeToMomentum"
# KFTrajectoryFitterESProducer   ---> Fitter = "KFFitterForRefitOutsideIn"
# KFTrajectorySmootherESProducer ---> Smoother = "KFSmootherForRefitOutsideIn"
# the propagator must be the same as the one used by the Fitter
#
globalCosmicMuons = cms.EDProducer("TracksToTrajectories",
                                   Tracks = cms.InputTag("globalCosmicMuons"),
                                   TrackTransformer = cms.PSet(TrackerRecHitBuilder = cms.string('WithTrackAngle'),
                                                               MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
                                                               RefitRPCHits = cms.bool(True),
                                                               Propagator = cms.string('SmartPropagatorAnyRK')
                                                               )
                                   )



MuAlGlobalCosmics = globalCosmicMuons.clone()
MuAlGlobalCosmics.Tracks = cms.InputTag("ALCARECOMuAlGlobalCosmics","GlobalMuon")
