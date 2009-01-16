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
                                   Type = cms.string("GlobalCosmicMuonsForAlignment"),
                                   Tracks = cms.InputTag("globalCosmicMuons"),
                                   TrackTransformer = cms.PSet(TrackerRecHitBuilder = cms.string('WithTrackAngle'),
                                                               MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
                                                               RefitRPCHits = cms.bool(True),
    															# muon station to be skipped
    															SkipStation		= cms.int32(-1),
    															# muon muon wheel to be skipped
															    #SkipDTWheel		= cms.int32(-1),
															    # muon muon wheel to be kept ( keeps plus and minus)  
															    KeepDTWheel		= cms.int32(-1000),
    	 														# PXB = 1, PXF = 2, TIB = 3, TID = 4, TOB = 5, TEC = 6
 															    TrackerSkipSystem	= cms.int32(-1),

    															# layer, wheel, or disk depending on the system
    															TrackerSkipSection	= cms.int32(-1),
                                                               )
                                   )



MuAlGlobalCosmics = globalCosmicMuons.clone()
MuAlGlobalCosmics.Tracks = cms.InputTag("ALCARECOMuAlGlobalCosmics","GlobalMuon")
