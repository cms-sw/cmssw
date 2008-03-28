import FWCore.ParameterSet.Config as cms

offlinePrimaryVerticesFromCTFTracks = cms.EDProducer("PrimaryVertexProducer",
    PVSelParameters = cms.PSet(
        maxDistanceToBeam = cms.double(0.02),
        minVertexFitProb = cms.double(0.01)
    ),
    verbose = cms.untracked.bool(False),
    algorithm = cms.string('AdaptiveVertexFitter'),
    TkFilterParameters = cms.PSet(
        maxNormalizedChi2 = cms.double(5.0),
        minSiliconHits = cms.int32(7),
        maxD0Significance = cms.double(5.0),
        minPt = cms.double(0.0),
        minPixelHits = cms.int32(2)
    ),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    # label of tracks to be used
    TrackLabel = cms.InputTag("generalTracks"),
    useBeamConstraint = cms.bool(False),
    VtxFinderParameters = cms.PSet(
        minTrackCompatibilityToOtherVertex = cms.double(0.01),
        minTrackCompatibilityToMainVertex = cms.double(0.05),
        maxNbVertices = cms.int32(0)
    ),
    TkClusParameters = cms.PSet(
        zSeparation = cms.double(0.1)
    )
)


