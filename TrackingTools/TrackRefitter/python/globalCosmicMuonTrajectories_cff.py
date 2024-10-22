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
                                   TrackTransformer = cms.PSet(
					TrackerRecHitBuilder = cms.string('WithTrackAngle'),
					MuonRecHitBuilder = cms.string('MuonRecHitBuilder'),
					MTDRecHitBuilder = cms.string('MTDRecHitBuilder'),
					RefitRPCHits = cms.bool(True),
					# muon station to be skipped //also kills RPCs in that station
					SkipStationDT	= cms.int32(-999),
					SkipStationCSC	= cms.int32(-999),
					# muon muon wheel to be skipped
					    SkipWheelDT		= cms.int32(-999),
					# PXB = 1, PXF = 2, TIB = 3, TID = 4, TOB = 5, TEC = 6
					    TrackerSkipSystem	= cms.int32(-999),
					# layer, wheel, or disk depending on the system
					TrackerSkipSection	= cms.int32(-999),
				   )
                                )



MuAlGlobalCosmics = globalCosmicMuons.clone(
    Tracks = "ALCARECOMuAlGlobalCosmics:GlobalMuon"
)
