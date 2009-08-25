import FWCore.ParameterSet.Config as cms

offlinePrimaryVerticesFromCosmicTracks = cms.EDProducer("PrimaryVertexProducer",
    PVSelParameters = cms.PSet(
        maxDistanceToBeam = cms.double(0.05), ## 200 microns

        minVertexFitProb = cms.double(0.01) ## 1% vertex fit probability

    ),
    verbose = cms.untracked.bool(False),
    algorithm = cms.string('AdaptiveVertexFitter'),
    TkFilterParameters = cms.PSet(
        maxNormalizedChi2 = cms.double(5.0),
        minSiliconHits = cms.int32(7), ## hits > 7

        maxD0Significance = cms.double(5.0), ## keep most primary tracks

        minPt = cms.double(0.0), ## better for softish events

        minPixelHits = cms.int32(2) ## hits > 2

    ),
    beamSpotLabel = cms.InputTag("offlineBeamSpot"),
    # label of tracks to be used
    TrackLabel = cms.InputTag("ctfWithMaterialTracksP5"),
    useBeamConstraint = cms.bool(False),
    VtxFinderParameters = cms.PSet(
        ptCut = cms.double(0.0),
        vtxFitProbCut = cms.double(0.01), ## 1% vertex fit probability
	trackCompatibilityToSVcut = cms.double(0.01), ## 1%
        trackCompatibilityToPVcut = cms.double(0.05), ## 5%
        maxNbOfVertices = cms.int32(0) ## search all vertices in each cluster

    ),
    TkClusParameters = cms.PSet(
        algorithm   = cms.string("gap"),
        TkGapClusParameter = cms.PSet( 
            zSeparation = cms.double(0.1) ## 1 mm max separation betw. clusters
        )
    )
)


