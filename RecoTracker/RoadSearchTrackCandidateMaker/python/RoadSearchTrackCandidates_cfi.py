import FWCore.ParameterSet.Config as cms

rsTrackCandidates = cms.EDProducer("RoadSearchTrackCandidateMaker",
    NumHitCut = cms.int32(5),
    # Initial Error on vertex in cm
    InitialVertexErrorXY = cms.double(0.2),
    # Initial Error on vertex in cm
    CosmicSeedPt = cms.double(5.0),
    HitChi2Cut = cms.double(30.0),
    StraightLineNoBeamSpotCloud = cms.bool(False),
    MeasurementTrackerName = cms.string(''),
    MinimumChunkLength = cms.int32(7),
    TTRHBuilder = cms.string('WithTrackAngle'),
    CosmicTrackMerging = cms.bool(False),
    nFoundMin = cms.int32(4),
    # cloud module label
    CloudProducer = cms.InputTag("roadSearchClouds"),
    SplitMatchedHits = cms.bool(False)
)


