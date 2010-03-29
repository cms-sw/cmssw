import FWCore.ParameterSet.Config as cms

pixelVertices = cms.EDProducer("PrimaryVertexProducer",
    PVSelParameters = cms.PSet(
        maxDistanceToBeam = cms.double(2), ## 2 cms in case there is no beamspot (ONLY FOR STARTUP!)

        minVertexFitProb = cms.double(0.01) ## 1% vertex fit probability

    ),
    verbose = cms.untracked.bool(False),
    algorithm = cms.string('AdaptiveVertexFitter'), ## 100 is for when the beamspot is not well known (ONLY FOR STARTUP)
    TkFilterParameters = cms.PSet(
        maxNormalizedChi2 = cms.double(100.0),
        minSiliconHits = cms.int32(2), ## hits > 2 - for when the beamspot is not well known (ONLY FOR STARTUP)

        maxD0Significance = cms.double(100.0), ## 100 is for when the beamspot is not well known (ONLY FOR STARTUP)

        minPt = cms.double(0.0), ## better for softish events

        minPixelHits = cms.int32(2) ## hits > 2

    ),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    # label of tracks to be used
    TrackLabel = cms.InputTag("pixelTracks"),
    useBeamConstraint = cms.bool(True),  # default for long term data and MC prod
    VtxFinderParameters = cms.PSet(
        ptCut = cms.double(0.0),
        vtxFitProbCut = cms.double(0.01), ## 1% vertex fit probability
	trackCompatibilityToSVcut = cms.double(0.01), ## 1%
        trackCompatibilityToPVcut = cms.double(0.05), ## 5%
        maxNbOfVertices = cms.int32(0) ## search all vertices in each cluster

    ),
    TkClusParameters = cms.PSet(
        zSeparation = cms.double(1) ## 1 cm max separation betw. clusters

    )
)


